import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# ---------------- 1. 기본 설정 ----------------
N_CHANNELS = 40
N_SLOTS = 10000
INTERFERENCE_THRESHOLD = 0.5
MAX_DELAY = 2        # 0,1,2 슬롯 기다림
K_HISTORY = 5
N_SAMPLES = 4000


def generate_interference_map(n_slots=N_SLOTS, n_channels=N_CHANNELS, rng=None):
    """시간×채널 간섭 맵 생성 (0~1)."""
    if rng is None:
        rng = np.random.default_rng()
    base = rng.uniform(0, 1, size=(n_slots, n_channels))
    # 특정 채널(10~15) 항상 조금 더 시끄럽게
    for ch in range(n_channels):
        if 10 <= ch <= 15:
            base[:, ch] += 0.3
    # 중앙부(약 5초 부근)에 전체 간섭 피크
    t = np.linspace(0, 1, n_slots)
    time_boost = 0.4 * np.exp(-((t - 0.5) ** 2) / 0.02)
    base += time_boost[:, None]
    return np.clip(base, 0, 1)


def send_packet(slot, channel, interference_map):
    """해당 슬롯/채널에서 패킷 전송 성공 여부."""
    return interference_map[slot, channel] < INTERFERENCE_THRESHOLD


# ---------------- 2. AFH 시뮬레이터 ----------------
def simulate_AFH(interference_map,
                 n_slots=N_SLOTS,
                 n_channels=N_CHANNELS,
                 window=50,
                 bad_threshold=0.4):
    """
    간단 AFH:
    - 최근 window 동안 에러율이 bad_threshold 초과인 채널은 제외
    - 남은 채널을 라운드로빈으로 사용
    """
    used_channels = np.ones(n_channels, dtype=bool)
    error_counts = np.zeros(n_channels, dtype=int)
    tx_counts = np.zeros(n_channels, dtype=int)

    successes = 0
    failures = 0
    history = []
    rr_idx = 0

    for slot in range(n_slots):
        active_ch = np.where(used_channels)[0]
        if len(active_ch) == 0:
            used_channels[:] = True
            active_ch = np.arange(n_channels)

        ch = active_ch[rr_idx % len(active_ch)]
        rr_idx += 1

        ok = send_packet(slot, ch, interference_map)
        tx_counts[ch] += 1
        if ok:
            successes += 1
        else:
            failures += 1
            error_counts[ch] += 1

        history.append((ch, ok))
        if len(history) > window:
            old_ch, old_ok = history.pop(0)
            if not old_ok:
                error_counts[old_ch] -= 1
            tx_counts[old_ch] -= 1

        with np.errstate(divide='ignore', invalid='ignore'):
            recent_err = np.where(tx_counts > 0,
                                  error_counts / np.maximum(tx_counts, 1),
                                  0.0)
        used_channels = recent_err < bad_threshold

    return successes, failures

# ---------------- 3. 현실적인 MLP 학습 데이터 ----------------
def build_realistic_training_data(interference_map,
                                  rng,
                                  k_history=K_HISTORY,
                                  n_samples=N_SAMPLES,
                                  n_channels=N_CHANNELS,
                                  max_delay=MAX_DELAY):
    """
    입력 X: 최근 k_history 슬롯 간섭 평균(40) + 직전 슬롯 위험 채널 플래그(40)
    라벨 y_ch, y_delay:
      같은 과거 상태에서 (채널, delay=0~max_delay)를 실제로 보낸다고 가정하고,
      '처음 성공한 조합'을 정답으로 사용
    """
    n_slots = interference_map.shape[0]
    X_list, y_ch_list, y_delay_list = [], [], []

    max_start = n_slots - k_history - (max_delay + 1)
    starts = rng.integers(0, max_start, size=n_samples)

    for s in starts:
        # 입력 피처
        hist = interference_map[s:s + k_history, :]
        feat_mean = hist.mean(axis=0)
        last_slot = interference_map[s + k_history - 1, :]
        bad_flag = (last_slot > 0.7).astype(float)
        feat = np.concatenate([feat_mean, bad_flag])

        best_ch = None
        best_delay = None

        # 현실적 라벨: 실제로 보낸다고 가정하고, 처음 성공한 (채널, delay)
        for ch in range(n_channels):
            for delay in range(max_delay + 1):
                target_slot = s + k_history + delay
                if target_slot >= n_slots:
                    continue
                ok = send_packet(target_slot, ch, interference_map)
                if ok:
                    best_ch = ch
                    best_delay = delay
                    break
            if best_ch is not None:
                break

        if best_ch is not None:
            X_list.append(feat)
            y_ch_list.append(best_ch)
            y_delay_list.append(best_delay)

    X = np.array(X_list)
    y_ch = np.array(y_ch_list, dtype=int)
    y_delay = np.array(y_delay_list, dtype=int)
    return X, y_ch, y_delay

# ---------------- 4. MLP 시뮬레이터들 ----------------
def simulate_MLP_channel_only(interference_map, mlp_ch,
                              k_history=K_HISTORY,
                              n_channels=N_CHANNELS):
    """MLP가 채널만 예측, delay=0(바로 전송) 고정."""
    successes = 0
    failures = 0
    n_slots = interference_map.shape[0]

    slot = k_history
    while slot < n_slots:
        hist = interference_map[slot - k_history:slot, :]
        feat_mean = hist.mean(axis=0)
        last_slot = interference_map[slot - 1, :]
        bad_flag = (last_slot > 0.7).astype(float)
        feat = np.concatenate([feat_mean, bad_flag])[None, :]

        ch = int(mlp_ch.predict(feat)[0])
        target_slot = slot

        ok = send_packet(target_slot, ch, interference_map)
        if ok:
            successes += 1
        else:
            failures += 1

        slot += 1

    return successes, failures


def simulate_MLP_channel_timing(interference_map, mlp_ch, mlp_delay,
                                k_history=K_HISTORY,
                                n_channels=N_CHANNELS,
                                max_delay=MAX_DELAY):
    """MLP가 채널+delay를 함께 예측."""
    successes = 0
    failures = 0
    n_slots = interference_map.shape[0]

    slot = k_history
    while slot < n_slots - max_delay:
        hist = interference_map[slot - k_history:slot, :]
        feat_mean = hist.mean(axis=0)
        last_slot = interference_map[slot - 1, :]
        bad_flag = (last_slot > 0.7).astype(float)
        feat = np.concatenate([feat_mean, bad_flag])[None, :]

        delay = int(mlp_delay.predict(feat)[0])
        ch = int(mlp_ch.predict(feat)[0])
        target_slot = slot + delay

        ok = send_packet(target_slot, ch, interference_map)
        if ok:
            successes += 1
        else:
            failures += 1

        slot += 1

    return successes, failures

# ---------------- 5. 시뮬레이션 1회 (seed 하나) ----------------
def run_once(seed):
    # 1) 랜덤 환경 생성
    rng = np.random.default_rng(seed)
    interference = generate_interference_map(rng=rng)

    # 2) 학습 데이터 생성
    X, y_ch, y_delay = build_realistic_training_data(interference, rng)
    X_train, X_test, ych_tr, ych_te = train_test_split(
        X, y_ch, test_size=0.2, random_state=0
    )
    _, _, ydel_tr, ydel_te = train_test_split(
        X, y_delay, test_size=0.2, random_state=0
    )

    # 3) MLP 학습
    mlp_ch = MLPClassifier(hidden_layer_sizes=(64, 64),
                           activation='relu',
                           max_iter=300,
                           random_state=0)
    mlp_delay = MLPClassifier(hidden_layer_sizes=(32, 32),
                              activation='relu',
                              max_iter=300,
                              random_state=1)
    mlp_ch.fit(X_train, ych_tr)
    mlp_delay.fit(X_train, ydel_tr)

    # 4) AFH / MLP 시뮬레이션
    afh_succ, afh_fail = simulate_AFH(interference)
    mlp_ch_succ, mlp_ch_fail = simulate_MLP_channel_only(interference, mlp_ch)
    mlp_ct_succ, mlp_ct_fail = simulate_MLP_channel_timing(interference, mlp_ch, mlp_delay)

    # 5) 성공률
    afh_rate    = afh_succ    / (afh_succ    + afh_fail)
    mlp_ch_rate = mlp_ch_succ / (mlp_ch_succ + mlp_ch_fail)
    mlp_ct_rate = mlp_ct_succ / (mlp_ct_succ + mlp_ct_fail)

    # 6) 정확도
    ch_acc  = mlp_ch.score(X_test, ych_te)
    del_acc = mlp_delay.score(X_test, ydel_te)

    # 7) 개수 + 퍼센트 출력
    print(f"[seed {seed}] AFH              : 성공 {afh_succ:5d} / 시도 {afh_succ+afh_fail:5d} ({afh_rate*100:6.3f}%)")
    print(f"[seed {seed}] MLP(채널만)      : 성공 {mlp_ch_succ:5d} / 시도 {mlp_ch_succ+mlp_ch_fail:5d} ({mlp_ch_rate*100:6.3f}%)")
    print(f"[seed {seed}] MLP(채널+타이밍) : 성공 {mlp_ct_succ:5d} / 시도 {mlp_ct_succ+mlp_ct_fail:5d} ({mlp_ct_rate*100:6.3f}%)")
    print(f"[seed {seed}] AI 정확도        : 채널 {ch_acc*100:6.3f}%, 타이밍 {del_acc*100:6.3f}%")
    print("-" * 60)

    return afh_rate, mlp_ch_rate, mlp_ct_rate, ch_acc, del_acc


# ---------------- 6. 여러 seed에 대해 반복 실행 ----------------
if __name__ == "__main__":
    for s in [0, 1, 2, 3, 4]:
        run_once(s)
