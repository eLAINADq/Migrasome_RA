#!/usr/bin/env python3

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.io import mmread
from scipy import stats
from scipy.stats import false_discovery_control
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
import os

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False

BASE_DIR = "/Users/dqy/PythonProject/lungcell_RA"
EXPORT_DIR = os.path.join(BASE_DIR, "export")
H5AD_PATH = os.path.join(BASE_DIR, "lungcell_mus_DQ.h5ad")

DC_ACTIVATION_GENES = [
    'Cd80', 'Cd86', 'Cd40', 'Cd83', 'Cd274', 'Pdcd1lg2',
    'H2-Aa', 'H2-Ab1', 'H2-Eb1', 'H2-DMa', 'H2-DMb1', 'H2-Ob',
    'Cd74', 'Ciita', 'Tap1', 'Tap2', 'Tapbp', 'B2m',
    'Psmb8', 'Psmb9', 'Psmb10', 'Ctss',
    'Ccr7', 'Fscn1', 'Lamp3', 'Marcksl1', 'Socs2',
    'Relb', 'Nfkb1', 'Nfkb2', 'Irf4', 'Irf8', 'Batf3',
    'Il12a', 'Il12b', 'Il6', 'Tnf', 'Il1b',
    'Ccl17', 'Ccl22', 'Cxcl9', 'Cxcl10',
]


def build_anndata():
    counts = mmread(os.path.join(EXPORT_DIR, "counts.mtx")).T.tocsr()
    genes = pd.read_csv(os.path.join(EXPORT_DIR, "genes.csv"))["x"].values
    barcodes = pd.read_csv(os.path.join(EXPORT_DIR, "barcodes.csv"))["x"].values
    metadata = pd.read_csv(os.path.join(EXPORT_DIR, "metadata.csv"), index_col=0)

    adata = ad.AnnData(X=counts)
    adata.obs_names = barcodes
    adata.var_names = genes
    adata.obs = metadata

    umap = pd.read_csv(os.path.join(EXPORT_DIR, "umap.csv"), index_col=0)
    adata.obsm["X_umap"] = umap.values

    pca = pd.read_csv(os.path.join(EXPORT_DIR, "pca.csv"), index_col=0)
    adata.obsm["X_pca"] = pca.values

    adata.write_h5ad(H5AD_PATH)
    return adata


def run_ora(gene_list, gene_sets, background):
    gene_set = set(gene_list)
    bg_set = set(background)
    N = len(bg_set)
    n = len(gene_set & bg_set)

    results = []
    for pathway, genes in gene_sets.items():
        pathway_set = set(genes)
        M = len(pathway_set & bg_set)
        k = len(gene_set & pathway_set)
        if k == 0:
            continue

        pval = stats.hypergeom.sf(k - 1, N, M, n)
        results.append({
            'Description': pathway,
            'Count': k,
            'GeneRatio': f"{k}/{n}",
            'GeneRatio_value': k / n,
            'BgRatio': f"{M}/{N}",
            'pvalue': pval,
            'Genes': '/'.join(gene_set & pathway_set)
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df['p.adjust'] = false_discovery_control(df['pvalue'], method='bh')
        df = df.sort_values('p.adjust')
    return df


def plot_ora_dotplot(df, title, output_name, output_dir, top_n=20):
    df_plot = df.head(top_n).copy()
    df_plot = df_plot.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))

    size_min, size_max = df_plot['Count'].min(), df_plot['Count'].max()
    if size_max > size_min:
        sizes = 80 + (df_plot['Count'] - size_min) / (size_max - size_min) * 320
    else:
        sizes = [150] * len(df_plot)

    pval_transformed = -np.log10(df_plot['p.adjust'].clip(lower=1e-20))
    vmin, vmax = pval_transformed.min(), pval_transformed.max()

    if vmax - vmin < 1:
        vmin = max(0, vmin - 0.5)
        vmax = vmax + 0.5

    scatter = ax.scatter(
        df_plot['GeneRatio_value'], range(len(df_plot)),
        c=pval_transformed, s=sizes, cmap='RdYlBu_r',
        edgecolors='darkgray', linewidths=0.5, alpha=0.9,
        vmin=vmin, vmax=vmax
    )

    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['Description'], fontsize=9)
    ax.set_xlabel('GeneRatio', fontsize=12)
    ax.set_xlim(0, None)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    cbar = plt.colorbar(scatter, shrink=0.4, pad=0.02)
    cbar.set_label('-log10(p.adjust)', fontsize=10)

    if size_max > size_min:
        legend_vals = np.unique(np.round(np.linspace(size_min, size_max, 4)).astype(int))
    else:
        legend_vals = [int(size_min)]

    for val in legend_vals:
        if size_max > size_min:
            sz = 80 + (val - size_min) / (size_max - size_min) * 320
        else:
            sz = 150
        ax.scatter([], [], c='gray', s=sz, label=str(val), edgecolors='darkgray', linewidths=0.5)

    ax.legend(title='Count', loc='lower right', frameon=True, fontsize=8,
              labelspacing=1.2, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{output_name}.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f"{output_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_fibroblast(adata):
    output_dir = os.path.join(BASE_DIR, "figures_fibro")
    os.makedirs(output_dir, exist_ok=True)

    adata_fibro = adata[adata.obs['celltype_coarse'] == 'Fibroblast'].copy()

    if adata_fibro.X.max() > 50:
        sc.pp.normalize_total(adata_fibro, target_sum=1e4)
        sc.pp.log1p(adata_fibro)

    group1, group2 = 'CIA_KO', 'CIA_Flox'

    sc.tl.rank_genes_groups(adata_fibro, groupby='condition',
                            groups=[group1], reference=group2, method='wilcoxon')
    de_genes = sc.get.rank_genes_groups_df(adata_fibro, group=group1)

    sig_genes = de_genes[
        (abs(de_genes['logfoldchanges']) > 0.5) &
        (de_genes['pvals_adj'] < 0.05)
    ]['names'].tolist()

    de_genes.to_csv(os.path.join(output_dir, "DEG_fibro.csv"), index=False)

    kegg_sets = gp.get_library(name='KEGG_2019_Mouse')
    gobp_sets = gp.get_library(name='GO_Biological_Process_2021')

    bg_genes = adata_fibro.var_names.tolist()

    df_kegg = run_ora(sig_genes, kegg_sets, bg_genes)
    df_kegg.to_csv(os.path.join(output_dir, "KEGG_ORA_results.csv"), index=False)

    df_gobp = run_ora(sig_genes, gobp_sets, bg_genes)
    df_gobp.to_csv(os.path.join(output_dir, "GOBP_ORA_results.csv"), index=False)

    plot_ora_dotplot(df_kegg, f'Fibroblast cells — KEGG ({group1} vs {group2})',
                     'KEGG_ORA_dotplot', output_dir, top_n=20)

    plot_ora_dotplot(df_gobp, f'Fibroblast cells — GO BP ({group1} vs {group2})',
                     'GOBP_ORA_dotplot', output_dir, top_n=20)


def analyze_dendritic_cells(adata):
    output_dir = os.path.join(BASE_DIR, "figures_DC")
    os.makedirs(output_dir, exist_ok=True)

    dc_types = [
        'Conventional dendritic cell type 1',
        'Conventional dendritic cell type 2',
        'Plasmacytoid dendritic cell',
        'Proliferating dendritic cell'
    ]

    adata_dc = adata[adata.obs['celltype_fine'].isin(dc_types)].copy()
    adata_dc = adata_dc[adata_dc.obs['condition'].isin(['CIA_Flox', 'Ctrl'])].copy()

    if adata_dc.X.max() > 50:
        sc.pp.normalize_total(adata_dc, target_sum=1e4)
        sc.pp.log1p(adata_dc)

    available_genes = set(adata_dc.var_names)
    found_genes = [g for g in DC_ACTIVATION_GENES if g in available_genes]

    sc.tl.score_genes(adata_dc, gene_list=found_genes, score_name='DC_activation_score')

    group1, group2 = 'CIA_Flox', 'Ctrl'
    score_col = 'DC_activation_score'

    g1 = adata_dc.obs[adata_dc.obs['condition'] == group1][score_col]
    g2 = adata_dc.obs[adata_dc.obs['condition'] == group2][score_col]

    stat, pval = stats.mannwhitneyu(g1, g2, alternative='two-sided')

    results = [{
        'Pathway': 'DC_activation',
        'CIA_Flox_mean': g1.mean(),
        'CIA_Flox_std': g1.std(),
        'Ctrl_mean': g2.mean(),
        'Ctrl_std': g2.std(),
        'Diff': g1.mean() - g2.mean(),
        'pvalue': pval
    }]

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "DC_activation_score.csv"), index=False)

    plot_data = adata_dc.obs[['condition', score_col]].copy()
    plot_data = plot_data[plot_data['condition'].isin([group1, group2])]
    colors = {'CIA_Flox': '#E41A1C', 'Ctrl': '#377EB8'}

    if pval < 0.001:
        ptext = '***'
    elif pval < 0.01:
        ptext = '**'
    elif pval < 0.05:
        ptext = '*'
    else:
        ptext = 'ns'

    fig, ax = plt.subplots(figsize=(4, 5))
    sns.violinplot(data=plot_data, x='condition', y=score_col,
                   palette=colors, order=[group2, group1], ax=ax)
    sns.stripplot(data=plot_data, x='condition', y=score_col,
                  color='black', size=1, alpha=0.3, order=[group2, group1], ax=ax)

    ymax = plot_data[score_col].max()
    ax.plot([0, 1], [ymax * 1.15, ymax * 1.15], 'k-', linewidth=1)
    ax.text(0.5, ymax * 1.18, ptext, ha='center', fontsize=14, fontweight='bold')

    ax.set_xlabel('')
    ax.set_ylabel('DC Activation Score', fontsize=12)
    ax.set_title('Dendritic Cell Activation\n(CIA_Flox vs Ctrl)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "DC_activation_violin.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "DC_activation_violin.png"), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 5))
    sns.boxplot(data=plot_data, x='condition', y=score_col,
                palette=colors, order=[group2, group1], ax=ax, width=0.5)
    sns.stripplot(data=plot_data, x='condition', y=score_col,
                  color='black', size=2, alpha=0.5, order=[group2, group1], ax=ax)

    ymax = plot_data[score_col].max()
    ax.plot([0, 1], [ymax * 1.15, ymax * 1.15], 'k-', linewidth=1)
    ax.text(0.5, ymax * 1.18, ptext, ha='center', fontsize=14, fontweight='bold')

    ax.set_xlabel('')
    ax.set_ylabel('DC Activation Score', fontsize=12)
    ax.set_title('Dendritic Cell Activation\n(CIA_Flox vs Ctrl)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "DC_activation_boxplot.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "DC_activation_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    if not os.path.exists(H5AD_PATH):
        adata = build_anndata()
    else:
        adata = sc.read_h5ad(H5AD_PATH)

    analyze_fibroblast(adata)
    analyze_dendritic_cells(adata)
