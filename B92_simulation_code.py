# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 09:27:33 2025
Modified to B92 on 2025-10-28

Author: Harry (modified by assistant)
Date: 2025-10-28
"""

import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.stats import norm, binom
import time

# Quantum states (as 2D complex vectors)
# |0> = [1,0], |1> = [0,1], |+> = (1/sqrt2)[1,1], |-> = (1/sqrt2)[1,-1]

sqrt2 = np.sqrt(2)

psi0 = np.array([1.0, 0.0], dtype=complex)                # Alice state for bit 0
psi1 = np.array([1.0/sqrt2, 1.0/sqrt2], dtype=complex)    # Alice state for bit 1 (|+>)

# Vectors orthogonal to Alice's states:
# For psi1 = |+>, an orthogonal vector is |-> = (1/sqrt2)[1,-1]
phi0 = np.array([1.0/sqrt2, -1.0/sqrt2], dtype=complex)   # orthogonal to psi1
# For psi0 = |0>, an orthogonal vector is |1> = [0,1]
phi1 = np.array([0.0, 1.0], dtype=complex)                # orthogonal to psi0

def project_probability(proj_vec, state_vec):
    """Probability that projector onto proj_vec clicks for input state state_vec."""
    # assume proj_vec normalized
    amp = np.vdot(proj_vec, state_vec)  # inner product <proj|state>
    return np.abs(amp)**2

# ------------------ Helpers for classical message/key handling ------------------

def string_to_binary(string):
    binary_list = []
    for char in string:
        bin_char = format(ord(char), '08b')
        for bit in bin_char:
            binary_list.append(int(bit))
    return binary_list

def binary_to_string(binary_list):
    chars = []
    for i in range(0, len(binary_list), 8):
        chunk = binary_list[i:i+8]
        if len(chunk) < 8:
            # pad with zeros if incomplete last byte
            chunk = chunk + [0]*(8-len(chunk))
        char_int = int(''.join(map(str, chunk)), 2)
        chars.append(chr(char_int))
    return ''.join(chars)

def key_too_short(data, key):
    if len(key) < len(data):
        print('Error - key should be longer than the data')
        return 'Error!'
    return key[:len(data)]

def encryption(data, key):
    if len(data) != len(key):
        key = key_too_short(data, key)
    return [data[i] ^ key[i] for i in range(len(data))]  # XOR

def decryption(message, key):
    return encryption(message, key)

# ------------------ B92 classical/quantum simulation functions ------------------

def random_b92(length):
    """Prepare random Alice bits and choose random measurement choices for Bob."""
    alice_bits = random.randint(2, size=length)  # 0 or 1
    # Bob will randomly choose which projector to test on each incoming pulse:
    # choice '0' means test projector onto phi0 (orth to psi1) -> click => infer Alice sent psi0 (bit 0)
    # choice '1' means test projector onto phi1 (orth to psi0) -> click => infer Alice sent psi1 (bit 1)
    bob_choice = random.randint(2, size=length)
    return alice_bits, bob_choice

def encode_state(bit):
    """Return the 2-vector representing Alice's state for a given bit (0 or 1)."""
    return psi0 if bit == 0 else psi1

def bob_measure_one(projector_choice, incoming_state):
    """
    Bob chooses one projector to test (phi0 or phi1).
    If projector clicks (probability = |<phi|state>|^2), the result is 'conclusive' and
    Bob assigns the corresponding bit. Otherwise it's 'inconclusive' (None).
    """
    if projector_choice == 0:
        p = project_probability(phi0, incoming_state)
        if random.random() < p:
            # phi0 clicked -> Bob infers Alice did NOT send psi1 -> infers bit 0
            return 0
        else:
            return None
    else:
        p = project_probability(phi1, incoming_state)
        if random.random() < p:
            # phi1 clicked -> Bob infers Alice did NOT send psi0 -> infers bit 1
            return 1
        else:
            return None

def simulate_b92_no_eve(alice_bits, bob_choice):
    """Simulate sending states and Bob's measurements without Eve.
       Returns bob_results list (None for inconclusive, or 0/1 for conclusive)."""
    bob_results = []
    for i in range(len(alice_bits)):
        state = encode_state(alice_bits[i])
        result = bob_measure_one(bob_choice[i], state)
        bob_results.append(result)
    return bob_results

def simulate_b92_with_eve(alice_bits, bob_choice):
    """
    Eve intercept-resend:
    - Eve measures using the same single-projector test strategy as Bob and
      if she gets conclusive result, she resends the corresponding Alice state.
    - If Eve's measurement is inconclusive she resends a random Alice state (psi0 or psi1).
    Returns bob_results, eve_results
    """
    n = len(alice_bits)
    eve_choices = random.randint(2, size=n)  # Eve randomly chooses which projector to test
    eve_results = []
    bob_results = []
    for i in range(n):
        incoming = encode_state(alice_bits[i])
        # Eve measures
        eve_result = bob_measure_one(eve_choices[i], incoming)
        eve_results.append(eve_result)
        # Eve prepares a new state to send to Bob
        if eve_result == 0:
            resend_state = psi0
        elif eve_result == 1:
            resend_state = psi1
        else:
            # inconclusive for Eve -> resend random state (simple intercept-resend model)
            resend_state = psi0 if random.randint(2) == 0 else psi1

        # Bob measures the resent state
        bob_res = bob_measure_one(bob_choice[i], resend_state)
        bob_results.append(bob_res)

    return bob_results, eve_results

def create_key_b92(alice_bits, bob_results):
    """
    From B92, only positions where Bob had conclusive results (0 or 1) are kept.
    We produce: res_key (list of bits Bob has), indices_kept (list of original indices kept),
    counts for matched/mismatched relative to Alice.
    """
    res_key = []
    indices = []
    count_true = 0
    count_false = 0
    for i, br in enumerate(bob_results):
        if br is not None:
            res_key.append(br)
            indices.append(i)
            if br == alice_bits[i]:
                count_true += 1
            else:
                count_false += 1
    total = count_true + count_false
    accuracy = count_true / (len(alice_bits))  # normalized by total sent (like your original)
    return {
        'res_key': res_key,
        'indices': indices,
        'len_key': len(res_key),
        'total_conclusive': total,
        'count_true': count_true,
        'count_false': count_false,
        'accuracy': accuracy
    }

def print_stats_b92(alice_bits, bob_results):
    info = create_key_b92(alice_bits, bob_results)
    n = len(alice_bits)
    total_conc = info['total_conclusive']
    count_false = info['count_false']
    count_true = info['count_true']
    QBER = count_false / total_conc if total_conc > 0 else 0
    print(f'Accuracy: {info["accuracy"]:.6f}')
    print('Total bits sent by Alice:', n)
    print('Conclusive (kept) outcomes:', total_conc)
    print('Matching bits (conclusive & correct):', count_true)
    print('Mismatched bits (conclusive & wrong):', count_false)
    print(f'QBER on the conclusive subset: {QBER:.6f}')
    print('---')

# ------------------ Monte Carlo QBER simulation for B92 with Eve ------------------

def compute_qber_b92(alice_bits, bob_results):
    """Compute QBER only on conclusive events (where Bob_results not None)."""
    indices = [i for i in range(len(bob_results)) if bob_results[i] is not None]
    if len(indices) == 0:
        return 0.0
    errors = sum(alice_bits[i] != bob_results[i] for i in indices)
    return errors / len(indices)

def monte_carlo_b92_with_eve(num_trials=500, key_length=5000, bins=30):
    qbers = []
    detections = 0
    print("\nRunning Monte Carlo B92 simulation with Eve...")
    start_time = time.time()

    for _ in range(num_trials):
        alice_bits, bob_choice = random_b92(key_length)
        bob_bits, eve_bits = simulate_b92_with_eve(alice_bits, bob_choice)
        qber = compute_qber_b92(alice_bits, bob_bits)
        qbers.append(qber)
        # detection threshold: if QBER (on conclusive subset) > 0.05 we say Eve detected
        if qber >= 0.05:
            detections += 1

    end_time = time.time()
    runtime = end_time - start_time
    mean_qber = np.mean(qbers)
    std_qber = np.std(qbers)
    detection_probability = detections / num_trials

    print("\n==============================")
    print("Monte Carlo B92 with Eve – QBER Summary")
    print(f"Trials: {num_trials}")
    print(f"Bits per trial: {key_length}")
    print(f"Mean QBER (conclusive subset): {mean_qber:.4f}")
    print(f"Standard Deviation: {std_qber:.4f}")
    print(f"Eve detected in {detections}/{num_trials} trials")
    print(f"Detection Probability: {detection_probability:.3f}")
    print(f"Runtime: {runtime:.2f} seconds")
    print("==============================")

    # Histogram + binomial & gaussian fits (approx)
    plt.figure(figsize=(8, 5))
    plt.hist(qbers, bins=bins, density=True, alpha=0.6, label='QBER Histogram')

    # Approx binomial fit: N ~ average conclusive outcomes per trial
    avg_conclusive = np.mean([sum(1 for r in simulate_b92_no_eve(*random_b92(key_length)) if r is not None) for _ in range(20)])
    N = max(1, int(avg_conclusive))
    p = mean_qber
    k_vals = np.arange(0, N+1)
    q_vals = k_vals / N
    pmf_vals = binom.pmf(k_vals, N, p)
    if len(q_vals) > 1:
        pdf_scaled = pmf_vals / (q_vals[1] - q_vals[0])
        plt.plot(q_vals, pdf_scaled, 'r--', linewidth=2, label='Binomial Fit (approx)')

    x = np.linspace(0, 1, 200)
    plt.plot(x, norm.pdf(x, mean_qber, std_qber), 'g-.', label='Gaussian Fit')

    # --- Adaptive x-axis and auto-scaled y-axis ---
    min_x = max(0, min(qbers) - 0.05)
    max_x = min(1, max(qbers) + 0.05)
    plt.xlim(min_x, max_x)

    # Auto-scale y-axis based on histogram + fits
    all_y = []
    # histogram heights
    counts, edges = np.histogram(qbers, bins=bins, density=True)
    all_y.extend(counts)
    # binomial fit
    if len(q_vals) > 1:
        all_y.extend(pdf_scaled)
    # gaussian fit
    all_y.extend(norm.pdf(x, mean_qber, std_qber))
    plt.ylim(0, max(all_y)*1.1)  # add 10% margin on top

    plt.xlabel('QBER (on conclusive bits)')
    plt.ylabel('Probability Density')
    plt.title('B92 QBER Distribution with Eve (approx fits)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        'qbers': qbers,
        'mean_qber': mean_qber,
        'std_qber': std_qber,
        'detection_probability': detection_probability,
        'runtime': runtime
    }

# ------------------ Main interactive block ------------------

if __name__ == "__main__":
    while True:
        try:
            length = int(input("Enter the number of bits to send: "))
            if length <= 0:
                raise ValueError
            break
        except ValueError:
            print("Please enter a valid positive integer.")

    eve_input = input("Include Eve in the simulation? (yes/no): ").strip().lower()
    include_eve = eve_input in ["yes", "y"]

    message = input("Enter the message Alice wants to send: ")
    binary_message = string_to_binary(message)

    alice_bits, bob_choice = random_b92(length)

    if not include_eve:
        bob_results = simulate_b92_no_eve(alice_bits, bob_choice)
        print("\n--- B92 Simulation without Eve ---")
        print_stats_b92(alice_bits, bob_results)
        key_info = create_key_b92(alice_bits, bob_results)
        key = key_info['res_key']
        print("Key length (conclusive bits):", len(key))
    else:
        bob_results, eve_results = simulate_b92_with_eve(alice_bits, bob_choice)
        print("\n--- B92 Simulation with Eve (intercept-resend) ---")
        print_stats_b92(alice_bits, bob_results)
        key_info = create_key_b92(alice_bits, bob_results)
        key = key_info['res_key']
        print("Key length (conclusive bits):", len(key))

        run_mc = input("\nRun Monte Carlo QBER analysis with Eve? (yes/no): ").strip().lower()
        if run_mc in ["yes", "y"]:
            try:
                num_trials = int(input("Enter number of Monte Carlo trials (e.g. 500): "))
            except ValueError:
                num_trials = 500
            key_len = length
            monte_carlo_b92_with_eve(num_trials=num_trials, key_length=key_len)

    # If the key is shorter than the message, key_too_short will warn
    key_for_message = key_too_short(binary_message, key)
    if key_for_message == 'Error!':
        # abort encryption step (or pad key randomly if you prefer)
        print("Encryption aborted due to insufficient key length.")
    else:
        encrypted_message = encryption(binary_message, key_for_message)
        decrypted_message = decryption(encrypted_message, key_for_message)
        decrypted_string = binary_to_string(decrypted_message)

        print("\n--- Message Transmission ---")
        print("Original message:", message)
        print("Decrypted message:", decrypted_string)
