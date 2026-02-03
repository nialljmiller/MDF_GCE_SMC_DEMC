# Parameter Reference

Complete documentation of all parameters in `bulge_pcard.txt` for the MDF_GCE_SMC_DEMC pipeline.

## Table of Contents

- [Infall Parameters](#infall-parameters)
- [Star Formation Parameters](#star-formation-parameters)
- [Initial Mass Function](#initial-mass-function)
- [Supernova Parameters](#supernova-parameters)
- [Galaxy Parameters](#galaxy-parameters)
- [Yield Tables](#yield-tables)
- [Observational Data](#observational-data)
- [GA Configuration](#ga-configuration)
- [Loss Configuration](#loss-configuration)
- [Output Configuration](#output-configuration)

---

## Infall Parameters

These parameters control the two-infall model for gas accretion.

### `sigma_2_list`

**Description**: Mass ratio of the second to the first infall episode.

**Units**: Dimensionless

**Typical Range**: [0.1, 5.0]

**Physical Interpretation**:
- `σ₂ < 1`: First infall dominates (larger early burst)
- `σ₂ > 1`: Second infall dominates (larger late burst)
- `σ₂ = 0`: Single infall model (first infall only)
- `σ₂ >> 1`: Approximates second-infall-only scenario

**Notes**: In the OMEGA+ `exp_infall` setup with A1=A2=-1, the total accreted mass M_tot is split as:
- M₁ = M_tot / (1 + σ₂)
- M₂ = σ₂ × M₁

**References**: Spitoni et al. (2019, A&A, 632, A58); Matteucci et al. (2024)

---

### `tmax_1_list`

**Description**: Time of first infall onset after the Big Bang.

**Units**: Gyr

**Typical Range**: [0.005, 0.1]

**Physical Interpretation**: Controls when the initial gas collapse begins. Very early times (0.005 Gyr) correspond to rapid primordial collapse; later times allow for pre-enrichment.

**Notes**: 
- t₁ = 0.01 Gyr corresponds to ~10 Myr after Big Bang
- Bulge stars are among the oldest in the Galaxy (>12 Gyr), requiring early onset

**References**: Chiappini et al. (2001); Ballero et al. (2007, A&A, 467, 943)

---

### `tmax_2_list`

**Description**: Time of second infall onset after the Big Bang.

**Units**: Gyr

**Typical Range**: [0.1, 10.0]

**Physical Interpretation**: 
- t₂ ~ 2 Gyr: "Thin disk" scenario (~11 Gyr ago)
- t₂ ~ 5 Gyr: "Thick disk" scenario (~8 Gyr ago)
- Small t₂ - t₁ gap: Rapid two-phase formation
- Large t₂ - t₁ gap: Distinct formation epochs

**Notes**: The second infall is often associated with Gaia-Enceladus-Sausage merger event.

**References**: Spitoni et al. (2019); Helmi et al. (2018, Nature, 563, 85)

---

### `infall_timescale_1_list`

**Description**: e-folding timescale (τ₁) of the first infall episode.

**Units**: Gyr

**Typical Range**: [0.001, 0.1]

**Physical Interpretation**: Short timescales (0.001-0.01 Gyr) produce rapid, burst-like gas accretion consistent with violent relaxation in the early bulge. Longer timescales (>0.05 Gyr) indicate more gradual assembly.

**Notes**: Also denoted τ₁ in the literature. Controls the shape of the infall rate: M_dot ∝ exp(-t/τ)

**References**: Matteucci & Brocato (1990); Grieco et al. (2012)

---

### `infall_timescale_2_list`

**Description**: e-folding timescale (τ₂) of the second infall episode.

**Units**: Gyr

**Typical Range**: [0.1, 10.0]

**Physical Interpretation**: Controls whether the second gas accretion is bursty (short τ₂) or extended (long τ₂). Long timescales are consistent with secular evolution or continuous accretion.

**References**: Grisoni et al. (2020, MNRAS, 492, 2828)

---

## Star Formation Parameters

### `sfe_array`

**Description**: Star formation efficiency — fraction of gas converted to stars per unit time.

**Units**: Gyr⁻¹

**Typical Range**: [1.0, 20.0]

**Physical Interpretation**:
- SFE ~ 0.01-0.1: Diffuse environments, slow star formation
- SFE ~ 0.02-0.2: Typical bulge values
- SFE ~ 0.5-1.0: Dense starbursts
- SFE > 1.0: Extreme environments (rapid depletion)

**Notes**: Higher SFE produces faster chemical enrichment and earlier α-element plateau turnover.

**References**: 
- Leroy et al. (2013, AJ, 146, 19): 0.01-1 in dense environments
- Bigiel et al. (2008, AJ, 136, 2846): 0.002-0.1 typical
- Grisoni et al. (2020): ~0.02 for bulge

---

### `delta_sfe_array`

**Description**: Multiplicative change in SFE that occurs at the second infall time.

**Units**: Gyr⁻¹ (multiplicative factor)

**Typical Range**: [0.01, 0.85]

**Physical Interpretation**:
- ΔSFE < 1: Quenching at second infall (dilution scenario)
- ΔSFE = 1: No change in SFE
- ΔSFE > 1: Enhanced star formation at second infall

**Notes**: Negative effective delta (decrease) may represent post-merger dilution or feedback effects.

**References**: Spitoni et al. (2019); Chiappini et al. (2001)

---

## Initial Mass Function

### `imf_array`

**Description**: Initial mass function determining the distribution of stellar masses.

**Units**: Categorical

**Options**: 
- `'salpeter'`: Power-law dn/dm ∝ m^(-2.35)
- `'chabrier'`: Log-normal for M < 1 M☉, power-law above
- `'kroupa'`: Multi-segment power law

**Physical Interpretation**: The IMF determines the relative numbers of low-mass (long-lived) vs high-mass (short-lived, yield-producing) stars. Salpeter produces more massive stars relative to Chabrier/Kroupa.

**Notes**: Salpeter is standard for bulge chemical evolution; Chabrier may be more appropriate for metal-rich populations.

**References**:
- Salpeter (1955, ApJ, 121, 161)
- Chabrier (2003, PASP, 115, 763)
- Kroupa (2001, MNRAS, 322, 231)

---

### `imf_upper_limits`

**Description**: Upper mass cutoff for the IMF.

**Units**: M☉

**Typical Range**: [60, 130]

**Physical Interpretation**: Limits the most massive stars considered. Used as a proxy when yield sets don't include failed supernova treatment. Stars above this limit are assumed to collapse directly to black holes without enriching the ISM.

**Notes**: 
- ~40 M☉: Conservative limit for metal-rich environments
- ~120-150 M☉: Observed in young massive clusters
- Above ~130 M☉: Pair-instability regime

**References**: Weidner et al. (2010); Pignatari et al. (2023)

---

## Supernova Parameters

### `nb_array`

**Description**: Number of Type Ia supernovae per solar mass of stars formed.

**Units**: M☉⁻¹

**Typical Range**: [0.5e-3, 1.5e-3]

**Physical Interpretation**: Controls the integrated SNe Ia rate and thus the iron-peak element production. Higher values produce more Fe relative to α-elements.

**Default**: 0.02 (reference: Grisoni et al. 2020)

**References**:
- Maoz et al. (2014, ARA&A, 52, 107): ~1e-3
- Matteucci et al. (2021): 3e-4 to 3e-3 in bulge
- Kobayashi et al. (2020): 0.0005-0.005 for Fe-peak

---

### `sn1a_rates`

**Description**: Delay-time distribution (DTD) model for SNe Ia.

**Units**: Categorical

**Options**:
- `'power_law'`: t^(-1) DTD (standard)
- `'gauss'`: Gaussian prompt component
- `'exp'`: Exponential (single-degenerate scenario)

**Physical Interpretation**: Determines when SNe Ia explode relative to star formation. Power-law produces extended iron enrichment; Gaussian concentrates SNe Ia near star formation.

**References**:
- Greggio (2005, A&A, 441, 1055): power-law
- Strolger et al. (2004, ApJ, 613, 200): Gaussian
- Maoz et al. (2014): empirical DTD

---

### `sn1a_assumptions`

**Description**: SNe Ia yield tables.

**Units**: Categorical (filenames)

**Options**: `'sn1a_Gronow.txt'`, `'sn1a_shen.txt'`, etc.

**Notes**: Different yield sets predict different iron-peak ratios.

---

## Galaxy Parameters

### `mgal_values`

**Description**: Initial gas mass of the bulge.

**Units**: M☉

**Typical Range**: [1e9, 1e11]

**Physical Interpretation**: Total baryonic reservoir available for star formation. Affects the dilution of metals by fresh gas and the total stellar mass formed.

**Notes**: The Milky Way bulge mass is ~1-2 × 10¹⁰ M☉.

**References**: Valenti et al. (2013); Portail et al. (2017)

---

## Yield Tables

### `comp_array`

**Description**: Initial abundance files for infall material metallicity.

**Units**: Categorical (filenames)

**Options**: Various `iniab_output_feh_*.txt` files with [Fe/H] from -2.0 to +0.5

**Physical Interpretation**: Sets the metallicity of infalling gas. Not primordial by default; can represent pre-enriched accretion.

---

### `stellar_yield_assumptions`

**Description**: Stellar yield tables for AGB and massive stars.

**Units**: Categorical (filenames)

**Notes**: Controls α-element and iron-peak production from CCSNe and AGB stars.

---

## Observational Data

### `obs_file`

**Description**: Path to observed MDF data file.

**Format**: CSV/TSV with [Fe/H] bins and normalized counts

---

### `obs_age_data_target`

**Description**: Age-metallicity data source.

**Options**: `'joyce'` or `'bensby'`

---

## GA Configuration

### `population_size`

**Description**: Number of individuals per GA generation.

**Default**: 96

**Notes**: Should be divisible by number of CPU cores for efficient parallelization.

---

### `num_generations`

**Description**: Total number of GA generations.

**Default**: 100

---

### `mutation_probability`

**Description**: Probability of mutation per gene.

**Default**: 0.1

---

### `crossover_probability`

**Description**: Probability of crossover between parents.

**Default**: 0.7

---

### `gaussian_sigma_scale`

**Description**: Standard deviation for Gaussian mutations as fraction of parameter range.

**Default**: 0.02

---

### `perturbation_strength`

**Description**: Strength of perturbations applied to break duplicate individuals.

**Default**: 0.2

---

## Loss Configuration

### `mdf_vs_age_weight`

**Description**: Weight for combining MDF loss vs age-metallicity loss.

**Range**: [0.0, 1.0]

**Interpretation**:
- 1.0 = MDF fitting only
- 0.5 = Equal weighting
- 0.8 = 80% MDF + 20% age-[Fe/H]

---

### `obs_age_data_loss_metric`

**Description**: Loss metric for age-metallicity fitting.

**Options**: `'mae'`, `'rmse'`, `'rms'`, `'weighted_mae'`, `'weighted_rmse'`, `'huber_loss'`, `'log_likelihood'`, `'aic'`, `'bic'`, `'correlation'`, `'spearman_correlation'`

---

## Output Configuration

### `output_path`

**Description**: Directory for all output files.

**Default**: `'SMC_DEMC/'`

---

### `timesteps`

**Description**: Number of time steps in OMEGA+ simulation.

**Default**: 100

**Notes**: Higher values increase precision but slow computation. Convergence typically achieved by 100-500 steps.

**References**: Côté et al. (2016, ApJ, 825, 126)
