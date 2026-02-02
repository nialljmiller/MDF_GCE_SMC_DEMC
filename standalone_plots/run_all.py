    # standalone_plots/run_all.py
import argparse
import os
import numpy as np
from .io import load_combined_table

# standalone_plots/run_all.py
import argparse
from .io import load_combined_table
from .plot_mdf import plot_mdf
from .plot_mdf import plot_mdf_posterior

from .plot_alpha import plot_alpha_posterior, plot_alpha_age_posterior
from .plot_age import plot_age_posterior
from .plot_omni import plot_omni, plot_omni_posterior

from .plot_corner import make_corner, make_corner_wlit, make_corner_mcmc, summarize_categorical_posterior
from .weights import attach_posterior_weights, temp_calc
from .best_selection import *



def main():

    obs_data_dir = "data/"

    parser = argparse.ArgumentParser(
        description="Standalone plotting from combined_full_evaluation_table.pkl"
    )
    parser.add_argument(
        "input_path",
        help="Path to combined_full_evaluation_table.pkl or its parent directory"
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for plots (default: same dir as PKL)"
    )
    args = parser.parse_args()

    df = load_combined_table(args.input_path)




    #for loss_metric in ['ks', 'ensemble', 'wrmse', 'mae', 'mape', 'huber', 'cosine', 'log_cosh', 'EMD', 'fitness']:
    #for loss_metric in ['wrmse', 'mae', 'EMD', 'fitness']:
    for loss_metric in ['wrmse']:
        
        L = df[loss_metric]

        best_idx = stable_best_index(df, primary=loss_metric)
        best_row = df.loc[best_idx]

        print(f"Global best model index: {best_idx} (loss={best_row[loss_metric]:.6f})")

        # 2. Define the categorical columns to filter on (from plot_omni.py)
        cat_cols = []
        #cat_cols = ['comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx']
        #cat_cols = ['comp_idx']#, 'sy_idx']
        #cat_cols = ['sy_idx']
        
        # Check which of these columns actually exist in the DataFrame
        valid_cat_cols = [c for c in cat_cols if c in df.columns]


        # 3. Build the filter mask
        mask = pd.Series(True, index=df.index)
        print("Filtering posterior to the 'best-fit' solution set:")
        for col in valid_cat_cols:
            best_val = best_row[col]
            print(f"  -> {col} == {best_val}")
            mask = mask & (df[col] == best_val)

        filtered_df = df[mask].copy() # Use .copy() to avoid warnings
        print(f"Original df size: {len(df)}, Filtered df size: {len(filtered_df)}")
        df = filtered_df

        T =  10#temp_calc(df, loss_col=loss_metric, target_ess_frac=0.25, T_lo=1e-3, T_hi=1e9, iters=400)

        df = attach_posterior_weights(df, loss_col=loss_metric, temperature=T)

        outdir = args.input_path + '/' + loss_metric

        top_overlay=99999999
        os.makedirs(outdir, exist_ok=True)

        kwargs = {
            "bins1d": 20,
            "bins2d": 30,
            "smooth": 2.0,
            "cmap": "Blues",
            "point_size": 10.0,
            "point_alpha": 0.15,
        }
        #plot_alpha(df, outdir, top_overlay = top_overlay, loss_metric=loss_metric)
        #plot_alpha_posterior(df, outdir, loss_metric=loss_metric)
        #plot_alpha_age_posterior(df, outdir, loss_metric=loss_metric)

        #plot_age_posterior(df, outdir, loss_metric=loss_metric)
        #plot_age(df, outdir, top_overlay = top_overlay, loss_metric=loss_metric)
        
        #plot_mdf_posterior(df, outdir, obs_mdf_path=obs_data_dir + 'equal_weight_mdf.dat', loss_metric=loss_metric)
        #plot_mdf(df, outdir, obs_mdf_path=obs_data_dir + 'equal_weight_mdf.dat', top_overlay = top_overlay, loss_metric=loss_metric)
        


        #make_corner(df, outdir, loss_metric=loss_metric, **kwargs)
        make_corner_wlit(df, outdir, loss_metric=loss_metric, **kwargs)
        #make_corner_mcmc(df, outdir, loss_metric=loss_metric, **kwargs)

        cat_cols = ['comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx']
        cat_cols = [c for c in cat_cols if c in df.columns]

        #summarize_categorical_posterior(df,outdir,cat_cols=cat_cols,hdi_mass=0.68,bins1d=30,loss_metric=loss_metric)
        #plot_omni_posterior(df, outdir, obs_mdf_path=obs_data_dir + 'equal_weight_mdf.dat', loss_metric=loss_metric)
        #plot_omni(df, outdir, obs_mdf_path=obs_data_dir + 'equal_weight_mdf.dat', top_overlay = top_overlay, loss_metric=loss_metric)




if __name__ == "__main__":
    main()
