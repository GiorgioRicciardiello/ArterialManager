"""
Do the distributions of image-derived overlap features differ across diseases?
both feature-wise and feature-set-wise.
For each feature column (pixel-level overlap metrics):

Runs a Kruskal–Wallis test across disease outcomes

Computes effect size (ε², nonparametric ANOVA effect size)

Aggregates results into:

feature-level table

group-level (A/B/C/D) summary table

Applies FDR correction

Produces a publication-ready table
"""
from config.config import CONFIG
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from library.ImgClassifier.CreateDataset import make_ml_feature_table
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def plot_overlap_distributions_hue_outcome_with_stats(
    df: pd.DataFrame,
    outcome_col: str,
    overlap_A: pd.Series,
    overlap_B: pd.Series,
    overlap_C: pd.Series,
    overlap_D: pd.Series,
    kde: bool = True,
    figsize=(20, 4),
    bins=30,
        output_path:Path=None
):
    """
    Columns = overlap groups (A–D)
    Hue = outcome

    Legend of EACH subplot includes per-outcome:
    mean, median, std, IQR.
    """

    # Build tidy dataframe
    plot_df = pd.DataFrame({
        "outcome": df[outcome_col],
        "Intensity overlap (A)": overlap_A,
        "Structural overlap (B)": overlap_B,
        "Channel balance (C)": overlap_C,
        "Information similarity (D)": overlap_D,
    })

    overlap_cols = plot_df.columns[1:]

    fig, axes = plt.subplots(
        1, len(overlap_cols),
        figsize=figsize,
        sharey=True
    )

    if len(overlap_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, overlap_cols):

        # Plot distributions
        sns.histplot(
            data=plot_df,
            x=col,
            hue="outcome",
            bins=bins,
            kde=kde,
            stat="density",
            common_norm=False,
            alpha=0.35,
            ax=ax,
            legend=False  # we will build a custom legend
        )

        # ---------- Global stats (title) ----------
        vals_all = plot_df[col].dropna()
        g_mean = vals_all.mean()
        g_median = vals_all.median()
        g_std = vals_all.std()
        g_iqr = vals_all.quantile(0.75) - vals_all.quantile(0.25)

        ax.set_title(
            f"{col}\n"
            f"μ={g_mean:.2f}, med={g_median:.2f}, σ={g_std:.2f}, IQR={g_iqr:.2f}",
            fontsize=11
        )

        # ---------- Per-outcome stats (legend) ----------
        legend_lines = []
        legend_labels = []

        outcomes = plot_df["outcome"].unique()
        palette = sns.color_palette(n_colors=len(outcomes))

        for outcome, color in zip(outcomes, palette):
            v = plot_df.loc[plot_df["outcome"] == outcome, col].dropna()
            if v.empty:
                continue

            mean = v.mean()
            median = v.median()
            std = v.std()
            iqr = v.quantile(0.75) - v.quantile(0.25)

            legend_lines.append(
                plt.Line2D([0], [0], color=color, lw=4)
            )
            legend_labels.append(
                f"{outcome} | μ={mean:.2f}, med={median:.2f}, "
                f"σ={std:.2f}, IQR={iqr:.2f}"
            )

        ax.legend(
            legend_lines,
            legend_labels,
            title="Outcome",
            fontsize=9,
            title_fontsize=10,
            loc="best",
            frameon=True
        )

        ax.set_xlabel("")
        ax.set_ylabel("Density")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path.joinpath(f'overlap_outcome_hist.png'), dpi=300)
    plt.show()

def _safe_mean(cols):
    return X_norm[cols].mean(axis=1) if len(cols) > 0 else 0


def _epsilon_squared(H, n, k):
    """
    Nonparametric effect size for Kruskal–Wallis.
    """
    return (H - k + 1) / (n - k)


def test_feature_distributions_across_outcomes(
    df: pd.DataFrame,
    outcome_col: str,
    feature_groups: dict
) -> pd.DataFrame:
    """
    Test whether feature distributions differ across outcomes
    using Kruskal–Wallis tests.

    Parameters
    ----------
    df : DataFrame
        Table with feature columns and outcome labels.
    outcome_col : str
        Column with disease labels.
    feature_groups : dict
        {"A": [cols], "B": [cols], ...}

    Returns
    -------
    DataFrame
        Feature-level summary table (paper-ready).
    """

    rows = []

    for group_name, cols in feature_groups.items():
        for col in cols:
            if col not in df.columns:
                continue

            # split values by outcome
            groups = [
                g[col].dropna().values
                for _, g in df.groupby(outcome_col)
                if g[col].notna().sum() > 1
            ]

            if len(groups) < 2:
                continue

            H, p = kruskal(*groups)

            n = df[col].notna().sum()
            k = len(groups)
            eps2 = _epsilon_squared(H, n, k)

            rows.append({
                "feature_group": group_name,
                "feature": col,
                "H_statistic": H,
                "p_value": p,
                "epsilon_squared": eps2,
                "n_samples": n,
                "n_groups": k
            })

    results = pd.DataFrame(rows)

    # FDR correction across ALL features
    results["p_fdr"] = multipletests(
        results["p_value"],
        method="fdr_bh"
    )[1]

    return results.sort_values("p_fdr")

def summarize_feature_groups(feature_results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate feature-level tests into group-level summaries.
    """

    summary = (
        feature_results
        .groupby("feature_group")
        .agg(
            n_features=("feature", "count"),
            n_significant=("p_fdr", lambda x: (x < 0.05).sum()),
            median_eps2=("epsilon_squared", "median"),
            max_eps2=("epsilon_squared", "max"),
            min_p_fdr=("p_fdr", "min")
        )
        .reset_index()
    )
    return summary.sort_values("median_eps2", ascending=False)



def fit_multinomial_discriminative_model(
    df_local,
    overlap_A,
    overlap_B,
    overlap_C,
    overlap_D,
    outcome_col
):
    """
    Fit a multinomial logistic regression to discriminate disease classes.
    Returns probabilities, coefficients, and global importance.
    """


    X_disc = pd.DataFrame({
        "Intensity_overlap": overlap_A,
        "Structural_overlap": overlap_B,
        "Channel_balance": overlap_C,
        "Information_similarity": overlap_D
    }, index=df_local.index)

    y = df_local[outcome_col]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_disc)

    clf = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    clf.fit(Xs, y)

    # Disease-specific coefficients
    coef_df = pd.DataFrame(
        clf.coef_,
        index=clf.classes_,
        columns=X_disc.columns
    )

    # Global importance (L2 norm across diseases)
    global_importance = pd.Series(
        np.linalg.norm(clf.coef_, axis=0),
        index=X_disc.columns
    ).sort_values(ascending=False)

    probs = pd.DataFrame(
        clf.predict_proba(Xs),
        index=df_local.index,
        columns=[f"P_{c}" for c in clf.classes_]
    )

    return {
        "model": clf,
        "scaler": scaler,
        "X_disc": X_disc,
        "probabilities": probs,
        "coefficients": coef_df,
        "global_importance": global_importance
    }



if __name__ == '__main__':
    # %% format table to feature matrix
    df = make_ml_feature_table(
        path_in=CONFIG.get('paths')['data'].joinpath(r"features_imgs\Imgfeatures.xlsx"),
        path_out=CONFIG.get('paths')['data'].joinpath(r"features_imgs\ImgfeaturesWide.xlsx"),
        id_col="cell",
        path_imgs=CONFIG.get('paths')['local_images_output'].joinpath('processed_overlap'),
        overwrite=False
    )
    dir_out = CONFIG.get("paths")['results'].joinpath('classification')
    dir_out.mkdir(exist_ok=True, parents=True)

    id_cols = ['cell', 'significant_overlap_absolute', 'img_path', 'outcome']
    outcome = 'outcome'  # name of the outcome column
    # %% Define feature columns
    features = [col for col in df.columns if (not df[col].isna().any()) and (col not in id_cols)]


    matched = []
    for f in features:
        matched.extend([c for c in df.columns if f in c])
    feature_cols = sorted(set(matched))

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    X_index = X.index
    df = df.loc[X_index]

    # normalize all metrics for composite score
    scaler_norm = MinMaxScaler()
    X_norm = pd.DataFrame(scaler_norm.fit_transform(X), columns=X.columns)

    # ----------------------------------------------------------------------
    # 1) COMPOSITE OVERLAP SCORE (OPTION 2)
    # ----------------------------------------------------------------------
    # A = intensity colocalization
    A_cols = [
        'weighted_overlap_red_absolute', 'weighted_overlap_green_absolute',
        'weighted_overlap_mean_absolute', 'sum_overlap_strength_absolute',
        'sum_heatmap_intensity', 'frac_heatmap_intensity',
        'Manders_M2_intensity', 'Manders_M1_intensity'
    ]

    # B = structural overlap
    B_cols = [
        'Dice_skeletons_intensity', 'Jaccard_skeletons_intensity',
        'Tanimoto_intensity', 'assd_skeleton_absolute', 'hausdorff_skeleton_absolute'
    ]

    # C = channel balance
    C_cols = [
        'green_red_ratio_absolute', 'red_green_ratio_absolute'
    ]

    # D = information metrics
    D_cols = [
        'mutual_information_absolute', 'cosine_similarity_absolute',
        'icq_absolute', 'bhattacharyya_coeff_absolute', 'hellinger_distance_absolute'
    ]



    overlap_A = _safe_mean([c for c in A_cols if c in X_norm])
    overlap_B = _safe_mean([c for c in B_cols if c in X_norm])
    overlap_C = _safe_mean([c for c in C_cols if c in X_norm])
    overlap_D = _safe_mean([c for c in D_cols if c in X_norm])

    plot_overlap_distributions_hue_outcome_with_stats(
        df=df,
        outcome_col=outcome,
        overlap_A=overlap_A,
        overlap_B=overlap_B,
        overlap_C=overlap_C,
        overlap_D=overlap_D
    )


    feature_groups = {
        "A_intensity": A_cols,
        "B_structural": B_cols,
        "C_balance": C_cols,
        "D_information": D_cols
    }

    feature_results = test_feature_distributions_across_outcomes(
        df=df,
        outcome_col=outcome,
        feature_groups=feature_groups
    )

    group_summary = summarize_feature_groups(feature_results)
    print(tabulate(group_summary,
                   headers=group_summary.columns,
                   tablefmt="github"))

    group_summary.to_excel(CONFIG.get('paths')['data'].joinpath(r"features_imgs\Imgfeatures_feature_groups.xlsx"))

    group_summary.to_csv(dir_out.joinpath(f"feature_groups_summary.csv"), index=False)
    # %%
    disc_results = fit_multinomial_discriminative_model(
        df_local=df,
        overlap_A=overlap_A,
        overlap_B=overlap_B,
        overlap_C=overlap_C,
        overlap_D=overlap_D,
        outcome_col=outcome
    )
    print(tabulate(disc_results.get('coefficients'), disc_results.get('coefficients').columns, 'github'))

    import numpy as np
    import matplotlib.pyplot as plt
    from tabulate import tabulate


    def plot_coefficient_heatmap(
            coef_df,
            *,
            figsize=(4.5, 2.5),
            cmap="RdBu_r",
            vlim=None,
            fmt="{:.2f}",
            fontsize=9,
            title=None,
            savepath=None
    ):
        """
        Minimalist heatmap for model coefficients (paper-ready).

        Parameters
        ----------
        coef_df : pd.DataFrame
            Rows = features, Columns = classes (or single column for binary).
        figsize : tuple
            Figure size in inches.
        cmap : str
            Diverging colormap centered at zero.
        vlim : float or None
            Symmetric color limit (+/- vlim). If None, inferred from max abs coef.
        fmt : str
            Format for coefficient text inside cells.
        fontsize : int
            Font size for cell annotations.
        title : str or None
            Optional figure title.
        savepath : str or Path or None
            If provided, saves figure (vector-friendly if .pdf/.svg).
        """

        # convert to numpy
        data = coef_df.values.astype(float)

        # symmetric color scale
        if vlim is None:
            vlim = np.nanmax(np.abs(data))

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=-vlim,
            vmax=vlim,
            aspect="auto"
        )

        # annotate cells
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                ax.text(
                    j,
                    i,
                    fmt.format(val),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="black" if abs(val) < 0.6 * vlim else "white"
                )

        # ticks & labels
        ax.set_xticks(np.arange(coef_df.shape[1]))
        ax.set_yticks(np.arange(coef_df.shape[0]))
        ax.set_xticklabels(coef_df.columns, rotation=30, ha="center")
        ax.set_yticklabels(coef_df.index)

        # minimalist styling
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

        # optional colorbar (very subtle)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(False)

        if title:
            ax.set_title(title, fontsize=10)

        fig.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")
            plt.close(fig)
        plt.show()

    # ------------------------------------------------------------
    # Example usage
    # ------------------------------------------------------------
    # coeffs = disc_results["coefficients"]   # pd.DataFrame
    # print(tabulate(coeffs, headers="keys", tablefmt="github", floatfmt=".2f"))
    plot_coefficient_heatmap(
        disc_results.get('coefficients'),
        figsize=(8.5, 6.5),
        fontsize=11,
        title="Discriminative Model Coefficients",
        savepath=None
    )

