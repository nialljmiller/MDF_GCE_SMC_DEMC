# Enhanced Physics Checks for Omega GCE GA

## Overview

This document describes the enhanced physics validation checks added to the omega GCE genetic algorithm code. These checks ensure that model outputs represent physically plausible galactic bulge evolution.

## New Physics Checks

### 1. Bulge Mass Check (`check_bulge_mass`)

**Purpose**: Validates that the final stellar mass is within reasonable bounds for a galactic bulge.

**Default Range**: 10⁹ to 10¹¹ M☉

**Rationale**: The Milky Way bulge has a mass of approximately 1-2 × 10¹⁰ M☉. This check ensures models don't produce unrealistically small or large stellar populations.

**Parameters**:
- `min_mass`: Minimum allowed stellar mass (default: 1e9 M☉)
- `max_mass`: Maximum allowed stellar mass (default: 1e11 M☉)
- `liberal`: If True, applies penalty instead of hard rejection

**Penalty Calculation**:
- Violation severity = (deviation from bound) / bound
- Penalty factor = 1 + 5 × violation_severity

### 2. Bulge Age Check (`check_bulge_age`)

**Purpose**: Ensures the system is old enough for a classical bulge.

**Default Minimum**: 10 Gyr

**Rationale**: Classical galactic bulges are old stellar populations that formed early in the universe. Systems younger than ~10 Gyr are inconsistent with observations of old bulges.

**Parameters**:
- `min_age_gyr`: Minimum allowed age in Gyr (default: 10.0)
- `liberal`: If True, applies penalty instead of hard rejection

**Penalty Calculation**:
- Violation severity = (min_age - actual_age) / min_age
- Penalty factor = 1 + 4 × violation_severity

### 3. Gas Fraction Check (`check_gas_fraction`)

**Purpose**: Validates that the final gas fraction is reasonable for an evolved system.

**Default Maximum**: 50% (0.5)

**Rationale**: Old bulges have converted most of their gas into stars. Very high gas fractions indicate incomplete or unrealistic evolution. The default threshold of 50% is conservative to allow for various evolutionary scenarios.

**Parameters**:
- `max_gas_fraction`: Maximum allowed gas fraction (default: 0.5)
- `liberal`: If True, applies penalty instead of hard rejection

**Calculation**:
- gas_fraction = gas_mass / (gas_mass + stellar_mass)

**Penalty Calculation**:
- Violation severity = (actual_fraction - max_fraction) / max_fraction
- Penalty factor = 1 + 3 × violation_severity

### 4. Star Formation History Peak Time Check (`check_sfh_peak_time`)

**Purpose**: Ensures the star formation rate peaks early, as expected for classical bulges.

**Default Maximum**: 3 Gyr

**Rationale**: Classical bulges formed through early, rapid star formation. Models where the SFR peaks late (> 3 Gyr) are inconsistent with bulge formation scenarios.

**Parameters**:
- `max_peak_time_gyr`: Maximum allowed time for SFR peak (default: 3.0 Gyr)
- `liberal`: If True, applies penalty instead of hard rejection

**Penalty Calculation**:
- Violation severity = (peak_time - max_time) / max_time
- Penalty factor = 1 + 2 × violation_severity

### 5. Mean Stellar Age Check (`check_mean_stellar_age`)

**Purpose**: Validates that the mass-weighted mean stellar age is old enough.

**Default Minimum**: 8 Gyr

**Rationale**: Bulge stellar populations should be predominantly old. A low mean age indicates too much recent star formation.

**Parameters**:
- `min_mean_age_gyr`: Minimum allowed mean stellar age (default: 8.0 Gyr)
- `liberal`: If True, applies penalty instead of hard rejection

**Calculation**:
- mean_age = Σ(mass_formed × stellar_age) / Σ(mass_formed)
- stellar_age = final_age - formation_time

**Penalty Calculation**:
- Violation severity = (min_age - actual_age) / min_age
- Penalty factor = 1 + 3 × violation_severity

## Integration

### Comprehensive Check Function

`check_model_physics(GCE_model, liberal=False)` runs all five checks and returns:
- `is_physical`: Boolean indicating if all checks pass
- `penalty_factor`: Combined penalty from all checks (multiplicative)

### Integration with Existing Checks

`apply_physics_penalty_with_model(...)` combines:
1. Existing MDF, alpha element, and age-metallicity checks
2. New model-level physics checks (if GCE_model is provided)

This function is called in `Gal_GA_PP.py` during fitness evaluation.

## Usage in GA Code

The physics checks are applied periodically during the genetic algorithm run, controlled by the `physical_constraints_freq` parameter:

```python
if self.physical_constraints_freq > 0:
    if self.physics_timer < self.physical_constraints_freq:
        self.physics_timer += 1
    else:
        self.physics_timer = 0
        penalty_factor = apply_physics_penalty_with_model(
            primary_loss_value, 
            MDF_x_data, MDF_y_data, 
            alpha_arrs, 
            age_x_data, age_y_data,
            GCE_model=GCE_model
        )
        primary_loss_value *= penalty_factor
```

## Customization

All check functions accept parameters to customize thresholds:

```python
# Example: Stricter gas fraction requirement
is_phys, penalty = check_gas_fraction(model, liberal=True, max_gas_fraction=0.2)

# Example: Require older mean age
is_phys, penalty = check_mean_stellar_age(model, liberal=True, min_mean_age_gyr=10.0)
```

To modify defaults globally, edit the function signatures in `physical_constraints.py`.

## Testing

Run the test script to verify all checks:

```bash
cd /home/ubuntu/MDF_GCE_SMC_DEMC
python3.11 test_physics_checks.py
```

Expected output:
- All individual checks should execute without errors
- Penalty factors should be reasonable (< 10 for realistic models)
- Combined penalty reflects cumulative violations

## References

- Milky Way bulge mass: ~1-2 × 10¹⁰ M☉ (McWilliam & Zoccali 2010)
- Bulge ages: > 10 Gyr (Zoccali et al. 2003, Bensby et al. 2017)
- Classical bulge formation: Early, rapid star formation (Kormendy & Kennicutt 2004)

## Author

N. Miller (2025)

## Changelog

### 2025-01-XX
- Added five new model-level physics checks
- Integrated with existing physics validation framework
- Created comprehensive test suite
- Updated `Gal_GA_PP.py` to use enhanced checks

