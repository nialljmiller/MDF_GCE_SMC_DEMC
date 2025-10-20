#!/usr/bin/env python3
"""
Test script for new physics checks in physical_constraints.py
"""
import sys
sys.path.append('/home/ubuntu/MDF_GCE_SMC_DEMC')
from JINAPyCEE import omega_plus
import numpy as np
from physical_constraints import (
    check_bulge_mass,
    check_bulge_age,
    check_gas_fraction,
    check_sfh_peak_time,
    check_mean_stellar_age,
    check_model_physics,
    apply_physics_penalty_with_model
)

print("=" * 70)
print("Testing New Physics Checks for Omega GCE GA")
print("=" * 70)

# Create a test model with reasonable parameters
print("\n1. Creating test omega_plus model...")
kwargs = {
    'mgal': 1e10,
    'dt': 1e7,
    'tend': 13e9,
    'sfe': 25.0,
    'table': 'yield_tables/agb_and_massive_stars_nugrid_MESAonly_fryer12delay.txt',
    'sn1a_table': 'yield_tables/sn1a_t86.txt',
    'print_off': True
}

try:
    model = omega_plus.omega_plus(**kwargs)
    print("   ✓ Model created successfully")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    sys.exit(1)

# Test individual physics checks
print("\n2. Testing individual physics checks...")

# Test bulge mass check
print("\n   a) Bulge mass check:")
try:
    is_phys, penalty = check_bulge_mass(model, liberal=True)
    m_stellar = model.inner.history.m_locked[-1]
    print(f"      Final stellar mass: {m_stellar:.2e} Msun")
    print(f"      Is physical: {is_phys}, Penalty: {penalty:.3f}")
    if is_phys and penalty < 2.0:
        print("      ✓ PASS")
    else:
        print("      ⚠ WARNING: High penalty or rejection")
except Exception as e:
    print(f"      ✗ FAIL: {e}")

# Test bulge age check
print("\n   b) Bulge age check:")
try:
    is_phys, penalty = check_bulge_age(model, liberal=True)
    age_final = model.inner.history.age[-1] / 1e9
    print(f"      Final age: {age_final:.2f} Gyr")
    print(f"      Is physical: {is_phys}, Penalty: {penalty:.3f}")
    if is_phys and penalty < 2.0:
        print("      ✓ PASS")
    else:
        print("      ⚠ WARNING: High penalty or rejection")
except Exception as e:
    print(f"      ✗ FAIL: {e}")

# Test gas fraction check
print("\n   c) Gas fraction check:")
try:
    is_phys, penalty = check_gas_fraction(model, liberal=True)
    gas_mass = np.sum(model.inner.ymgal[-1])
    stellar_mass = model.inner.history.m_locked[-1]
    gas_frac = gas_mass / (gas_mass + stellar_mass)
    print(f"      Final gas mass: {gas_mass:.2e} Msun")
    print(f"      Final stellar mass: {stellar_mass:.2e} Msun")
    print(f"      Gas fraction: {gas_frac:.3f}")
    print(f"      Is physical: {is_phys}, Penalty: {penalty:.3f}")
    if is_phys and penalty < 2.0:
        print("      ✓ PASS")
    else:
        print("      ⚠ WARNING: High penalty or rejection")
except Exception as e:
    print(f"      ✗ FAIL: {e}")

# Test SFH peak time check
print("\n   d) SFH peak time check:")
try:
    is_phys, penalty = check_sfh_peak_time(model, liberal=True)
    sfr = np.array(model.inner.history.sfr_abs)
    ages = np.array(model.inner.history.age) / 1e9
    peak_idx = np.argmax(sfr)
    peak_time = ages[peak_idx]
    print(f"      SFR peak time: {peak_time:.2f} Gyr")
    print(f"      Is physical: {is_phys}, Penalty: {penalty:.3f}")
    if is_phys and penalty < 2.0:
        print("      ✓ PASS")
    else:
        print("      ⚠ WARNING: High penalty or rejection")
except Exception as e:
    print(f"      ✗ FAIL: {e}")

# Test mean stellar age check
print("\n   e) Mean stellar age check:")
try:
    is_phys, penalty = check_mean_stellar_age(model, liberal=True)
    sfr = np.array(model.inner.history.sfr_abs)
    timesteps = np.array(model.inner.history.timesteps)
    ages = np.array(model.inner.history.age) / 1e9
    if len(sfr) > len(timesteps):
        sfr = sfr[:len(timesteps)]
    mass_formed = sfr * timesteps
    final_age = ages[-1]
    stellar_ages = final_age - ages[:len(timesteps)]
    mean_age = np.sum(mass_formed * stellar_ages) / np.sum(mass_formed)
    print(f"      Mean stellar age: {mean_age:.2f} Gyr")
    print(f"      Is physical: {is_phys}, Penalty: {penalty:.3f}")
    if is_phys and penalty < 2.0:
        print("      ✓ PASS")
    else:
        print("      ⚠ WARNING: High penalty or rejection")
except Exception as e:
    print(f"      ✗ FAIL: {e}")

# Test comprehensive model physics check
print("\n3. Testing comprehensive model physics check...")
try:
    is_phys, total_penalty = check_model_physics(model, liberal=True)
    print(f"   Overall is physical: {is_phys}")
    print(f"   Total penalty factor: {total_penalty:.3f}")
    if is_phys and total_penalty < 5.0:
        print("   ✓ PASS: Model passes all physics checks")
    else:
        print("   ⚠ WARNING: High total penalty")
except Exception as e:
    print(f"   ✗ FAIL: {e}")

# Test integration with existing physics checks
print("\n4. Testing integration with existing physics checks...")
try:
    # Get MDF and alpha data
    MDF_x, MDF_y = model.inner.plot_mdf(axis_mdf='[Fe/H]', sigma_gauss=0.1, norm=True, return_x_y=True)
    
    elements = ['[Si/Fe]', '[Ca/Fe]', '[Mg/Fe]', '[Ti/Fe]']
    alpha_arrs = []
    for el in elements:
        alpha_x, alpha_y = model.inner.plot_spectro(xaxis='[Fe/H]', yaxis=el, return_x_y=True)
        alpha_arrs.append([np.array(alpha_x), np.array(alpha_y)])
    
    age_x, age_y = model.inner.plot_spectro(xaxis='age', yaxis='[Fe/H]', return_x_y=True)
    
    # Apply combined physics penalty
    penalty = apply_physics_penalty_with_model(
        1.0,  # base loss
        MDF_x, MDF_y,
        alpha_arrs,
        age_x, age_y,
        GCE_model=model
    )
    
    print(f"   Combined penalty factor: {penalty:.3f}")
    if penalty < 10.0:
        print("   ✓ PASS: Combined physics checks work")
    else:
        print("   ⚠ WARNING: Very high combined penalty")
except Exception as e:
    print(f"   ✗ FAIL: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Testing complete!")
print("=" * 70)

