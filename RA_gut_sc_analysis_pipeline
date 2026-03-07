#!/usr/bin/env python
"""Single-cell RNA-seq analysis pipeline for RA gut mouse model (KO vs WT)."""

import os
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import celltypist
from celltypist import models
import gseapy as gp
import liana as li
from scipy import stats
import torch
import warnings

warnings.filterwarnings("ignore")
scvi.settings.seed = 42
sc.settings.verbosity = 2

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.titlesize": 9,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DATA_DIR = "./data"
OUTPUT_DIR = "./results"
REF_FILE = "./ref/lung_mus_ref.h5ad"
FIG_BASE = "./fig"

SAMPLES = {"KO": f"{DATA_DIR}/KO", "WT": f"{DATA_DIR}/WT"}

QC_PARAMS = dict(
    min_umi=500, min_genes=300, max_genes=5000,
    min_cells=3, max_mt_pct=5, max_hb_pct=1,
)

CELLTYPE_ORDER = [
    "Macrophage", "Monocyte", "DC", "Neutrophil", "Basophil",
    "T cell", "NK cell", "B cell", "Plasma cell",
    "AT2", "Club cell", "Ciliated cell", "Neuroendocrine",
    "Fibroblast", "Smooth muscle", "Endothelial", "Pericyte", "Adventitial cell",
]

COARSE_MAP = {
    "lung macrophage": "Macrophage", "alveolar macrophage": "Macrophage",
    "classical monocyte": "Monocyte", "non-classical monocyte": "Monocyte",
    "intermediate monocyte": "Monocyte",
    "CD4-positive, alpha-beta T cell": "T cell",
    "CD8-positive, alpha-beta T cell": "T cell",
    "regulatory T cell": "T cell", "mature NK T cell": "T cell", "T cell": "T cell",
    "natural killer cell": "NK cell",
    "B cell": "B cell", "plasma cell": "Plasma cell",
    "myeloid dendritic cell": "DC", "plasmacytoid dendritic cell": "DC",
    "dendritic cell": "DC",
    "neutrophil": "Neutrophil", "basophil": "Basophil",
    "pulmonary alveolar type 2 cell": "AT2", "club cell": "Club cell",
    "multiciliated columnar cell of tracheobronchial tree": "Ciliated cell",
    "pulmonary neuroendocrine cell": "Neuroendocrine",
    "fibroblast of lung": "Fibroblast",
    "pulmonary interstitial fibroblast": "Fibroblast",
    "bronchial smooth muscle cell": "Smooth muscle",
    "smooth muscle cell of the pulmonary artery": "Smooth muscle",
    "vein endothelial cell": "Endothelial",
    "endothelial cell of lymphatic vessel": "Endothelial",
    "adventitial cell": "Adventitial cell", "pericyte": "Pericyte",
}

COLORS = {
    "Macrophage": "#E69F9F", "Monocyte": "#F4B183", "DC": "#FFE699",
    "Neutrophil": "#C9A88C", "Basophil": "#F4CCCC",
    "T cell": "#9DC3E6", "NK cell": "#A9D08E", "B cell": "#C5A3CF",
    "Plasma cell": "#E6A8C8",
    "AT2": "#A8DBC5", "Club cell": "#7ECBB4", "Ciliated cell": "#B4B0D8",
    "Neuroendocrine": "#F5B87A",
    "Fibroblast": "#A6A6A6", "Smooth muscle": "#D0CECE",
    "Endothelial": "#8FAADC", "Pericyte": "#92C591",
    "Adventitial cell": "#F9CBC8",
}

MARKERS = {
    "Macrophage": ["Cd68", "Marco", "Mrc1"],
    "Monocyte": ["Ly6c2", "Ccr2", "Cd14"],
    "DC": ["Itgax", "Cd74", "H2-Aa"],
    "Neutrophil": ["S100a8", "S100a9"],
    "T cell": ["Cd3d", "Cd3e"],
    "NK cell": ["Ncr1", "Nkg7"],
    "B cell": ["Cd79a", "Ms4a1"],
    "Plasma cell": ["Jchain", "Xbp1"],
    "AT2": ["Sftpc", "Sftpa1"],
    "Club cell": ["Scgb1a1", "Scgb3a1"],
    "Ciliated cell": ["Foxj1"],
    "Fibroblast": ["Col1a1", "Col1a2"],
    "Smooth muscle": ["Acta2", "Tagln"],
    "Endothelial": ["Pecam1", "Cdh5"],
    "Pericyte": ["Pdgfrb", "Rgs5"],
}

FIB_INFLAM = ["Il1a", "Il1b", "Ccl2", "Il6", "Ccl7"]
FIB_FIBROSIS = [
    "Fbn1", "Col3a1", "Postn", "Col4a1", "Tgfb2", "Col1a2", "Tgfb3",
    "Mmp2", "Timp1", "Pdgfb", "Pdgfa", "Acta2", "Serpine1", "Loxl2",
    "Mmp9", "Fn1", "Thbs1", "Mmp14", "Cxcl1", "Cxcl12", "Ptgs2",
    "Cxcl2", "Nos2",
]
DC_INFLAM = [
    "Il6", "Il1b", "Il12a", "Il12b", "Tnf", "Cxcl9", "Cxcl10",
    "Ccl17", "Ccl22", "Nos2", "Il23a",
]
DC_COSTIM = [
    "Cd80", "Cd86", "Cd83", "Cd40", "Cd274", "H2-Aa", "H2-Ab1",
    "H2-Eb1", "Icam1", "Cd70",
]
DC_ACTIVATION = [
    "Cd80", "Cd86", "Cd83", "Cd70", "Cd274", "Pdcd1lg2",
    "H2-Aa", "H2-Ab1", "H2-Eb1", "H2-DMb1",
    "Ciita", "Ccr7", "Relb", "Nfkb1", "Socs2",
    "Il12a", "Il12b", "Il23a", "Il15", "Il1b", "Il6",
    "Cxcl9", "Cxcl10", "Ccl5",
    "Irf4", "Irf8", "Stat4",
    "Tlr2", "Tlr4", "Tlr9", "Ripk2",
]
FIB_ACTIVATION = [
    "Col1a1", "Col1a2", "Col3a1", "Fn1", "Acta2", "Postn",
    "Tgfb1", "Tgfb2", "Mmp2", "Mmp9", "Timp1", "Serpine1",
    "Loxl2", "Fbn1", "Thbs1",
]

USE_GPU = torch.backends.mps.is_available()
ACCELERATOR = "mps" if USE_GPU else "cpu"

for d in [OUTPUT_DIR, f"{FIG_BASE}/fig1", f"{FIG_BASE}/fig2",
          f"{FIG_BASE}/fig3", f"{FIG_BASE}/fig4"]:
    os.makedirs(d, exist_ok=True)


# ── Utilities ─────────────────────────────────────────────────

def gene_lookup(adata):
    return {g.lower(): g for g in adata.var_names}


def get_gene(var_lower, g):
    return var_lower.get(g.lower(), None)


def filter_genes(adata, genes):
    vl = gene_lookup(adata)
    return [vl.get(g.lower(), g) for g in genes if g.lower() in vl]


def to_log_norm(adata):
    ad = adata.copy()
    if "counts" in ad.layers:
        ad.X = ad.layers["counts"].copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    return ad


# ── Step 1: QC + Solo + scVI ─────────────────────────────────

def step1_qc_solo_scvi():
    print("[Step 1] QC + Solo + scVI")

    adata_list = []
    for name, path in SAMPLES.items():
        ad = sc.read_10x_mtx(path, var_names="gene_symbols", cache=True)
        ad.obs["sample"] = name
        ad.obs["condition"] = name
        ad.obs_names = [f"{name}_{bc}" for bc in ad.obs_names]
        ad.var_names_make_unique()
        adata_list.append(ad)

    adata = sc.concat(adata_list, join="outer")
    adata.obs_names_make_unique()

    adata.obs["total_counts"] = np.array(adata.X.sum(axis=1)).flatten()
    adata = adata[adata.obs["total_counts"] >= QC_PARAMS["min_umi"]].copy()

    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    adata.var["hb"] = adata.var_names.str.contains("^Hb[^(p)]", regex=True)
    hb_genes = adata.var_names[adata.var["hb"]].tolist()

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "hb"],
                               percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_genes(adata, min_cells=QC_PARAMS["min_cells"])

    adata = adata[adata.obs.n_genes_by_counts >= QC_PARAMS["min_genes"]].copy()
    adata = adata[adata.obs.n_genes_by_counts < QC_PARAMS["max_genes"]].copy()
    adata = adata[adata.obs.pct_counts_mt < QC_PARAMS["max_mt_pct"]].copy()
    adata = adata[adata.obs.pct_counts_hb < QC_PARAMS["max_hb_pct"]].copy()

    hb_in_data = [g for g in hb_genes if g in adata.var_names]
    if hb_in_data:
        adata = adata[:, ~adata.var_names.isin(hb_in_data)].copy()

    adata.layers["counts"] = adata.X.copy()

    # Solo doublet detection
    scores_list, preds_list = [], []
    for sample in adata.obs["sample"].unique():
        adata_s = adata[adata.obs["sample"] == sample].copy()
        scvi.model.SCVI.setup_anndata(adata_s, layer="counts")
        vae = scvi.model.SCVI(adata_s, n_layers=2, n_latent=30, gene_likelihood="nb")
        vae.train(max_epochs=100, early_stopping=True, early_stopping_patience=10,
                  train_size=0.9, batch_size=128, accelerator=ACCELERATOR)
        solo = scvi.external.SOLO.from_scvi_model(vae)
        solo.train(max_epochs=100, early_stopping=True, early_stopping_patience=10,
                   accelerator=ACCELERATOR)
        preds = solo.predict()
        score = preds["doublet"] if "doublet" in preds.columns else preds.iloc[:, 0]
        pred = pd.Series(
            ["doublet" if s > 0.5 else "singlet" for s in score], index=score.index
        )
        score.index = adata_s.obs_names
        pred.index = adata_s.obs_names
        scores_list.append(score)
        preds_list.append(pred)

    adata.obs["doublet_score"] = pd.concat(scores_list).loc[adata.obs_names].values
    adata.obs["doublet_prediction"] = pd.concat(preds_list).loc[adata.obs_names].values
    adata = adata[adata.obs["doublet_prediction"] == "singlet"].copy()

    # scVI integration
    adata_hvg = adata.copy()
    sc.pp.normalize_total(adata_hvg, target_sum=1e4)
    sc.pp.log1p(adata_hvg)
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=3000, batch_key="sample",
                                flavor="seurat_v3", layer="counts")
    adata.var["highly_variable"] = adata_hvg.var["highly_variable"]

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="sample",
                                  continuous_covariate_keys=["pct_counts_mt"])
    model = scvi.model.SCVI(adata, n_layers=2, n_latent=30, gene_likelihood="nb")
    model.train(max_epochs=400, early_stopping=True, early_stopping_patience=20,
                train_size=0.9, batch_size=128, accelerator=ACCELERATOR)

    adata.obsm["X_scVI"] = model.get_latent_representation()
    adata.layers["scvi_normalized"] = model.get_normalized_expression(library_size=1e4)

    adata.write(f"{OUTPUT_DIR}/RA_gut_scVI.h5ad")
    model.save(f"{OUTPUT_DIR}/scVI_model", overwrite=True)
    print(f"  Cells: {adata.n_obs:,}, Genes: {adata.n_vars:,}")
    return adata


# ── Step 2: UMAP + Clustering ────────────────────────────────

def step2_umap_cluster(adata):
    print("[Step 2] UMAP + Leiden clustering")
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
    sc.tl.umap(adata, min_dist=0.3)
    sc.tl.leiden(adata, resolution=0.8)
    adata.write(f"{OUTPUT_DIR}/RA_gut_processed.h5ad")
    print(f"  Clusters: {adata.obs['leiden'].nunique()}")
    return adata


# ── Step 3: CellTypist ───────────────────────────────────────

def step3_celltypist(adata):
    print("[Step 3] CellTypist annotation")
    adata_ct = to_log_norm(adata)
    adata_ct.var_names = adata_ct.var_names.str.upper()
    adata_ct.var_names_make_unique()

    model = models.Model.load(model="Human_Lung_Atlas.pkl")
    pred = celltypist.annotate(adata_ct, model=model, majority_voting=True)
    adata.obs["celltype"] = pred.to_adata().obs["majority_voting"]
    adata.obs["celltype_conf"] = pred.to_adata().obs["conf_score"]
    return adata


# ── Step 4: scANVI with reference ────────────────────────────

def step4_scanvi(adata):
    print("[Step 4] scANVI annotation")

    adata_query = adata.copy()
    adata_query.obs["dataset"] = "query"
    adata_query.obs["cell_type"] = "Unknown"

    adata_ref = sc.read_h5ad(REF_FILE)
    adata_ref.obs["dataset"] = "reference"
    for col in ["cell_type", "cell_ontology_class", "free_annotation"]:
        if col in adata_ref.obs.columns:
            adata_ref.obs["cell_type"] = adata_ref.obs[col].astype(str)
            break

    # Harmonize gene names via lowercase matching
    adata_query.var["gene_lower"] = adata_query.var_names.str.lower()
    adata_ref.var["gene_lower"] = adata_ref.var_names.str.lower()
    query_map = dict(zip(adata_query.var["gene_lower"], adata_query.var_names))
    ref_map = dict(zip(adata_ref.var["gene_lower"], adata_ref.var_names))
    common = set(adata_query.var["gene_lower"]) & set(adata_ref.var["gene_lower"])

    if len(common) < 1000:
        raise ValueError(f"Too few common genes ({len(common)})")

    common = list(common)
    adata_query_sub = adata_query[:, [query_map[g] for g in common]].copy()
    adata_query_sub.var_names = common
    adata_ref_sub = adata_ref[:, [ref_map[g] for g in common]].copy()
    adata_ref_sub.var_names = common

    if "counts" in adata_query_sub.layers:
        adata_query_sub.X = adata_query_sub.layers["counts"].copy()
    if "counts" in adata_ref_sub.layers:
        adata_ref_sub.X = adata_ref_sub.layers["counts"].copy()

    merged = sc.concat([adata_ref_sub, adata_query_sub],
                       label="batch", keys=["ref", "query"])
    merged.obs_names_make_unique()
    merged.layers["counts"] = merged.X.copy()

    sc.pp.filter_genes(merged, min_cells=10)
    sc.pp.normalize_total(merged, target_sum=1e4)
    sc.pp.log1p(merged)
    sc.pp.highly_variable_genes(merged, n_top_genes=3000, batch_key="batch",
                                flavor="seurat_v3", layer="counts")

    scvi.model.SCVI.setup_anndata(merged, layer="counts", batch_key="batch")
    vae = scvi.model.SCVI(merged, n_layers=2, n_latent=30, gene_likelihood="nb")
    vae.train(max_epochs=200, early_stopping=True, early_stopping_patience=15,
              batch_size=128, accelerator=ACCELERATOR)

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        vae, adata=merged, unlabeled_category="Unknown", labels_key="cell_type"
    )
    scanvi_model.train(max_epochs=100, early_stopping=True,
                       early_stopping_patience=10, batch_size=128,
                       accelerator=ACCELERATOR)

    merged.obs["prediction"] = scanvi_model.predict()
    query_mask = merged.obs["dataset"] == "query"
    adata.obs["scanvi_celltype"] = merged.obs.loc[query_mask, "prediction"].values

    # Coarse mapping
    adata.obs["celltype_coarse"] = (
        adata.obs["scanvi_celltype"]
        .map(COARSE_MAP)
        .fillna(adata.obs["scanvi_celltype"])
    )

    existing = [ct for ct in CELLTYPE_ORDER if ct in adata.obs["celltype_coarse"].values]
    adata.obs["celltype_coarse"] = pd.Categorical(
        adata.obs["celltype_coarse"], categories=existing, ordered=True
    )

    adata.write(f"{OUTPUT_DIR}/RA_gut_final.h5ad")
    adata.obs.to_csv(f"{OUTPUT_DIR}/celltype_annotation.csv")
    print(f"  Cell types: {adata.obs['celltype_coarse'].nunique()}")
    return adata


# ── Figure 1: UMAP + Proportions ─────────────────────────────

def fig1(adata):
    print("[Fig 1] UMAP + Proportions")
    fig_dir = f"{FIG_BASE}/fig1"
    existing = [ct for ct in CELLTYPE_ORDER
                if ct in adata.obs["celltype_coarse"].values]
    var_lower = gene_lookup(adata)

    marker_genes = []
    for ct in existing:
        if ct in MARKERS:
            marker_genes.extend(MARKERS[ct])
    marker_genes_filtered = [var_lower.get(g.lower(), g) for g in marker_genes
                             if g.lower() in var_lower]

    # 1A: UMAP by cell type
    fig, ax = plt.subplots(figsize=(5, 4))
    for ct in existing:
        mask = adata.obs["celltype_coarse"] == ct
        if mask.sum() > 0:
            ax.scatter(adata.obsm["X_umap"][mask, 0], adata.obsm["X_umap"][mask, 1],
                       s=1, c=COLORS.get(ct, "#999"), label=ct,
                       alpha=0.75, edgecolors="none", rasterized=True)
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.legend(markerscale=5, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/Fig1A_UMAP.pdf", bbox_inches="tight"); plt.close()

    # 1B: UMAP by sample
    fig, ax = plt.subplots(figsize=(3.5, 3))
    for sample, color in [("WT", "#2166AC"), ("KO", "#B2182B")]:
        mask = adata.obs["sample"] == sample
        ax.scatter(adata.obsm["X_umap"][mask, 0], adata.obsm["X_umap"][mask, 1],
                   s=0.8, c=color, label=sample, alpha=0.6,
                   edgecolors="none", rasterized=True)
    ax.legend(markerscale=5, frameon=False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/Fig1B_UMAP_sample.pdf", bbox_inches="tight"); plt.close()

    # 1C: Dotplot
    adata_plot = to_log_norm(adata)
    sc.pl.dotplot(adata_plot, var_names=marker_genes_filtered,
                  groupby="celltype_coarse", categories_order=existing,
                  cmap="Reds", dot_max=0.5, standard_scale="var", show=False)
    plt.savefig(f"{fig_dir}/Fig1C_Dotplot.pdf", bbox_inches="tight"); plt.close()

    # 1D: Proportion bar
    prop = pd.crosstab(adata.obs["celltype_coarse"], adata.obs["sample"],
                       normalize="columns") * 100
    prop = prop.loc[[ct for ct in existing if ct in prop.index]]

    fig, ax = plt.subplots(figsize=(2.5, 4))
    bottom = np.zeros(2)
    for ct in prop.index:
        ax.bar(["WT", "KO"], prop.loc[ct], bottom=bottom, label=ct,
               color=COLORS.get(ct, "#999"), width=0.6, edgecolor="white", linewidth=0.3)
        bottom += prop.loc[ct].values
    ax.set_ylabel("Proportion (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/Fig1D_Proportion.pdf", bbox_inches="tight"); plt.close()

    # 1E: Proportion comparison
    fig, ax = plt.subplots(figsize=(4, 4))
    x = np.arange(len(prop.index))
    w = 0.35
    ax.barh(x - w / 2, prop["WT"], w, label="WT", color="#2166AC")
    ax.barh(x + w / 2, prop["KO"], w, label="KO", color="#B2182B")
    ax.set_xlabel("Proportion (%)")
    ax.set_yticks(x); ax.set_yticklabels(prop.index)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/Fig1E_Proportion_compare.pdf", bbox_inches="tight")
    plt.close()

    counts = pd.crosstab(adata.obs["celltype_coarse"], adata.obs["sample"])
    counts = counts.loc[[ct for ct in existing if ct in counts.index]]
    stats_df = counts.copy()
    stats_df["Total"] = stats_df.sum(axis=1)
    stats_df["WT_pct"] = prop["WT"]
    stats_df["KO_pct"] = prop["KO"]
    stats_df["Diff_pct"] = stats_df["KO_pct"] - stats_df["WT_pct"]
    stats_df.sort_values("Diff_pct", ascending=False).to_csv(
        f"{fig_dir}/celltype_proportion_stats.csv"
    )


# ── Figure 2: Heatmaps + FC Barplots + Violin ────────────────

def _make_heatmap(adata, celltype, genes, title, save_path, vmin=-1.5, vmax=1.5):
    vl = gene_lookup(adata)
    cells = adata[adata.obs["celltype_coarse"] == celltype].copy()
    if cells.n_obs == 0:
        return

    valid_genes, data = [], {"WT": [], "KO": []}
    for g in genes:
        gn = get_gene(vl, g)
        if gn and gn in cells.var_names:
            for sample in ["WT", "KO"]:
                sub = cells[cells.obs["sample"] == sample]
                if sub.n_obs > 0:
                    expr = sub[:, gn].X
                    data[sample].append(
                        expr.toarray().mean() if hasattr(expr, "toarray") else expr.mean()
                    )
                else:
                    data[sample].append(0)
            valid_genes.append(g)

    if not valid_genes:
        return

    df = pd.DataFrame(data, index=valid_genes)
    df_z = df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1), axis=0)
    df_z = df_z.fillna(0).clip(vmin, vmax)

    h = max(2.5, len(valid_genes) * 0.22)
    fig, ax = plt.subplots(figsize=(1.8, h))
    im = ax.imshow(df_z.values, cmap="RdBu_r", aspect="auto",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["WT", "KO"], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(valid_genes)))
    ax.set_yticklabels(valid_genes, fontsize=6, fontstyle="italic")
    ax.set_title(title, fontweight="bold", fontsize=9, pad=8)
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.08)
    cbar.set_label("Z-score", fontsize=6); cbar.ax.tick_params(labelsize=5)
    for i in range(len(valid_genes) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5)
    ax.axvline(0.5, color="white", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight"); plt.close()


def _calc_fc(adata, celltype, genes):
    vl = gene_lookup(adata)
    cells = adata[adata.obs["celltype_coarse"] == celltype]
    if cells.n_obs == 0:
        return None

    results = []
    for g in genes:
        gn = get_gene(vl, g)
        if gn is None or gn not in cells.var_names:
            continue
        wt = cells[cells.obs["sample"] == "WT", gn].X
        ko = cells[cells.obs["sample"] == "KO", gn].X
        wt_v = wt.toarray().flatten() if hasattr(wt, "toarray") else wt.flatten()
        ko_v = ko.toarray().flatten() if hasattr(ko, "toarray") else ko.flatten()
        fc = np.log2((ko_v.mean() + 0.01) / (wt_v.mean() + 0.01))
        _, pval = stats.mannwhitneyu(ko_v, wt_v, alternative="two-sided")
        results.append(dict(Gene=g, WT_mean=wt_v.mean(), KO_mean=ko_v.mean(),
                            log2FC=fc, pvalue=pval))
    return pd.DataFrame(results).sort_values("log2FC", ascending=True) if results else None


def _fc_barplot(df, title, save_path):
    if df is None or len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(4, max(3, len(df) * 0.3)))
    y = range(len(df))
    colors = ["#E64B35" if x > 0 else "#3C5488" for x in df["log2FC"]]
    ax.barh(y, df["log2FC"].values, color=colors, edgecolor="none", height=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Gene"].values, fontstyle="italic")
    for i, (_, row) in enumerate(df.iterrows()):
        star = ("***" if row["pvalue"] < 0.001 else
                "**" if row["pvalue"] < 0.01 else
                "*" if row["pvalue"] < 0.05 else "")
        if star:
            ha = "right" if row["log2FC"] < 0 else "left"
            offset = -0.02 if row["log2FC"] < 0 else 0.02
            ax.text(row["log2FC"] + offset, i, star,
                    va="center", ha=ha, fontsize=7, fontweight="bold")
    ax.set_xlabel("log2 Fold Change (KO/WT)")
    ax.set_title(title, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight"); plt.close()


def _activation_violin(adata, celltype, gene_list, score_name, title, fig_dir):
    COLOR_WT, COLOR_KO = "#6A9BD2", "#D94F4F"
    vl = gene_lookup(adata)
    cells = adata[adata.obs["celltype_coarse"] == celltype].copy()
    valid = [get_gene(vl, g) for g in gene_list if get_gene(vl, g) in adata.var_names]
    if not valid:
        return None

    sc.tl.score_genes(cells, gene_list=valid, score_name=score_name)
    scores = cells.obs[score_name].values
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    cells.obs[f"{score_name}_norm"] = scores_norm

    wt_s = cells.obs.loc[cells.obs["sample"] == "WT", f"{score_name}_norm"].values
    ko_s = cells.obs.loc[cells.obs["sample"] == "KO", f"{score_name}_norm"].values
    _, pval = stats.mannwhitneyu(wt_s, ko_s, alternative="two-sided")
    sig = ("***" if pval < 0.001 else "**" if pval < 0.01 else
           "*" if pval < 0.05 else "n.s.")

    fig, ax = plt.subplots(figsize=(3, 4))
    for pos, vals, color in [(0, wt_s, COLOR_WT), (1, ko_s, COLOR_KO)]:
        parts = ax.violinplot(vals, positions=[pos], showmeans=False,
                              showmedians=False, showextrema=False, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_edgecolor("black")
            pc.set_linewidth(0.5); pc.set_alpha(0.8)
        np.random.seed(42 + pos)
        jitter = np.random.normal(0, 0.06, size=len(vals))
        ax.scatter(pos + jitter, vals, s=1.5, c="black", alpha=0.3,
                   zorder=3, rasterized=True, marker="o", linewidths=0)
        med = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        ax.hlines(med, pos - 0.15, pos + 0.15, color="black", linewidth=2, zorder=4)
        ax.vlines(pos, q1, q3, color="black", linewidth=1.2, zorder=4)

    y_max = max(wt_s.max(), ko_s.max())
    by = y_max + 0.06
    ax.plot([0, 0, 1, 1], [by - 0.015, by, by, by - 0.015], color="black", linewidth=0.8)
    ax.text(0.5, by + 0.005, sig, ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WT", "KO"], fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{title} Score", fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.05, by + 0.13)
    plt.tight_layout()
    tag = "_".join(title.split())
    plt.savefig(f"{fig_dir}/Fig2_{tag}_violin.pdf", bbox_inches="tight"); plt.close()

    # Per-gene stats
    gene_stats = []
    for g in gene_list:
        gn = get_gene(vl, g)
        if gn is None or gn not in cells.var_names:
            continue
        wt_e = cells[cells.obs["sample"] == "WT", gn].X
        ko_e = cells[cells.obs["sample"] == "KO", gn].X
        wt_v = wt_e.toarray().flatten() if hasattr(wt_e, "toarray") else wt_e.flatten()
        ko_v = ko_e.toarray().flatten() if hasattr(ko_e, "toarray") else ko_e.flatten()
        _, p = stats.mannwhitneyu(wt_v, ko_v, alternative="two-sided")
        fc = np.log2((ko_v.mean() + 0.01) / (wt_v.mean() + 0.01))
        gene_stats.append(dict(Gene=g, WT_mean=wt_v.mean(), KO_mean=ko_v.mean(),
                               log2FC=fc, pvalue=p))
    return pd.DataFrame(gene_stats) if gene_stats else None


def fig2(adata):
    print("[Fig 2] Marker heatmaps + FC barplots + Activation violin")
    fig_dir = f"{FIG_BASE}/fig2"
    adata_norm = to_log_norm(adata)

    panels = [
        ("Fibroblast", FIB_INFLAM, "Inflammation markers\n(Fibroblast)"),
        ("Fibroblast", FIB_FIBROSIS, "Fibrosis markers\n(Fibroblast)"),
        ("DC", DC_INFLAM, "Inflammation markers\n(DC)"),
        ("DC", DC_COSTIM, "Costimulatory markers\n(DC)"),
    ]
    tags = ["Fib_inflam", "Fib_fibrosis", "DC_inflam", "DC_costim"]

    for (ct, genes, title), tag in zip(panels, tags):
        _make_heatmap(adata_norm, ct, genes, title,
                      f"{fig_dir}/Fig2_{tag}_heatmap")
        fc_df = _calc_fc(adata_norm, ct, genes)
        if fc_df is not None:
            _fc_barplot(fc_df, f"{ct} — {title.split(chr(10))[0]}",
                        f"{fig_dir}/Fig2_{tag}_FC")
            fc_df.to_csv(f"{fig_dir}/{tag}_FC.csv", index=False)

    dc_stats = _activation_violin(adata_norm, "DC", DC_ACTIVATION,
                                  "DC_Activation", "DC Activation", fig_dir)
    if dc_stats is not None:
        dc_stats.to_csv(f"{fig_dir}/DC_activation_stats.csv", index=False)

    fib_stats = _activation_violin(adata_norm, "Fibroblast", FIB_ACTIVATION,
                                   "Fibrosis", "Fibroblast Activation", fig_dir)
    if fib_stats is not None:
        fib_stats.to_csv(f"{fig_dir}/Fibroblast_activation_stats.csv", index=False)


# ── Figure 3: GSEA ───────────────────────────────────────────

def _run_gsea_prerank(adata, celltype, min_cells=30):
    cells = adata[adata.obs["celltype_coarse"] == celltype].copy()
    if cells.n_obs < min_cells:
        return None
    n_wt = (cells.obs["sample"] == "WT").sum()
    n_ko = (cells.obs["sample"] == "KO").sum()
    if n_wt < 10 or n_ko < 10:
        return None

    cells.obs["group"] = cells.obs["sample"].astype(str)
    sc.tl.rank_genes_groups(cells, groupby="group", groups=["KO"],
                            reference="WT", method="wilcoxon")
    result = sc.get.rank_genes_groups_df(cells, group="KO").dropna()
    result["score"] = (
        -np.log10(result["pvals"].clip(lower=1e-300)) *
        np.sign(result["logfoldchanges"])
    )
    ranked = result.set_index(result["names"].str.upper())["score"]
    ranked = ranked[~ranked.index.duplicated(keep="first")].sort_values(ascending=False)

    try:
        res = gp.prerank(rnk=ranked, gene_sets="Reactome_2022", outdir=None,
                         min_size=10, max_size=500, permutation_num=100,
                         seed=42, verbose=False)
        return res.res2d
    except Exception:
        return None


def _gsea_bubble(df, title, save_path, top_n=15):
    if df is None or len(df) == 0:
        return
    df = df.head(top_n).copy()
    df["Term_clean"] = df["Term"].str.replace(r"R-HSA-\d+\s*", "", regex=True).str[:50]
    fdr = np.clip(pd.to_numeric(df["FDR q-val"], errors="coerce").fillna(1.0).values
                  .astype(np.float64), 1e-10, 1.0)
    df["-log10(FDR)"] = -np.log10(fdr)
    nes_abs = np.abs(pd.to_numeric(df["NES"], errors="coerce").fillna(0).values
                     .astype(np.float64))
    df = df.sort_values("-log10(FDR)", ascending=True)

    fig, ax = plt.subplots(figsize=(5, max(3, len(df) * 0.28)))
    ax.scatter(df["-log10(FDR)"].values, range(len(df)),
               s=nes_abs * 40, c=df["-log10(FDR)"].values, cmap="Reds",
               edgecolors="black", linewidths=0.3, alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Term_clean"].values, fontsize=6)
    ax.set_xlabel("-log10(FDR)")
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", bbox_inches="tight"); plt.close()


def fig3(adata):
    print("[Fig 3] GSEA (Reactome)")
    fig_dir = f"{FIG_BASE}/fig3"
    adata_norm = to_log_norm(adata)

    inflam_kw = [
        "interleukin", "cytokine", "inflamm", "interferon", "chemokine",
        "nfkb", "nf-kb", "toll", "tnf", "il-", "immune", "innate",
        "adaptive", "antigen", "mhc", "complement", "neutrophil",
        "macrophage", "dendritic",
    ]

    for ct, label in [("DC", "DC"), ("Fibroblast", "Fibroblast"),
                      ("AT2", "AT2"), ("Club cell", "Club")]:
        gsea_df = _run_gsea_prerank(adata_norm, ct)
        if gsea_df is None:
            continue

        gsea_df.to_csv(f"{fig_dir}/{label}_GSEA_full.csv", index=False)
        gsea_df["NES"] = pd.to_numeric(gsea_df["NES"], errors="coerce")
        gsea_df["FDR q-val"] = pd.to_numeric(gsea_df["FDR q-val"], errors="coerce")
        down = gsea_df[(gsea_df["NES"] < 0) & (gsea_df["FDR q-val"] < 0.25)]

        mask = down["Term"].str.lower().apply(
            lambda x: any(kw in x for kw in inflam_kw)
        )
        filtered = down[mask] if mask.any() else down.head(15)
        filtered = filtered.sort_values("NES", ascending=True)

        if len(filtered) > 0:
            _gsea_bubble(filtered, f"{ct}: Downregulated Pathways (KO vs WT)",
                         f"{fig_dir}/Fig3_{label}_down_reactome")
            filtered.to_csv(f"{fig_dir}/{label}_down_reactome.csv", index=False)


# ── Figure 4: LIANA Cell-Cell Communication ───────────────────

def fig4(adata):
    print("[Fig 4] LIANA cell-cell communication")
    fig_dir = f"{FIG_BASE}/fig4"
    adata_norm = to_log_norm(adata)

    adata_norm.var_names = [g.upper() for g in adata_norm.var_names]
    adata_norm.var_names_make_unique()
    adata_norm.obs["celltype"] = adata_norm.obs["celltype_coarse"].astype(str)

    min_cells = 30
    results = {}
    for sample_name in ["WT", "KO"]:
        ad = adata_norm[adata_norm.obs["sample"] == sample_name].copy()
        ct_counts = ad.obs["celltype"].value_counts()
        valid = ct_counts[ct_counts >= min_cells].index.tolist()
        ad = ad[ad.obs["celltype"].isin(valid)].copy()

        li.mt.rank_aggregate(ad, groupby="celltype", expr_prop=0.1,
                             verbose=True, use_raw=False)
        res = ad.uns["liana_res"].copy()
        res["sample"] = sample_name
        results[sample_name] = res
        res.to_csv(f"{fig_dir}/LIANA_{sample_name}.csv", index=False)

    wt_res, ko_res = results["WT"], results["KO"]

    # Differential interactions
    for res in [wt_res, ko_res]:
        res["interaction"] = (res["source"] + " -> " + res["target"] + " | " +
                              res["ligand_complex"] + ":" + res["receptor_complex"])

    merged = pd.merge(
        wt_res[["interaction", "source", "target", "ligand_complex",
                "receptor_complex", "magnitude_rank"]],
        ko_res[["interaction", "magnitude_rank"]],
        on="interaction", how="outer", suffixes=("_WT", "_KO")
    )
    merged["magnitude_rank_WT"] = merged["magnitude_rank_WT"].fillna(1.0)
    merged["magnitude_rank_KO"] = merged["magnitude_rank_KO"].fillna(1.0)
    merged["diff"] = merged["magnitude_rank_WT"] - merged["magnitude_rank_KO"]
    merged["abs_diff"] = merged["diff"].abs()
    merged = merged.sort_values("abs_diff", ascending=False)
    merged.to_csv(f"{fig_dir}/LIANA_differential.csv", index=False)

    # 4A: Interaction count heatmaps
    for name, res in [("WT", wt_res), ("KO", ko_res)]:
        sig = res[res["magnitude_rank"] < 0.05]
        if len(sig) == 0:
            sig = res[res["magnitude_rank"] < 0.1]
        counts = sig.groupby(["source", "target"]).size().reset_index(name="count")
        if len(counts) == 0:
            continue
        pivot = counts.pivot(index="source", columns="target", values="count").fillna(0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(pivot, cmap="Reds", annot=True, fmt=".0f", annot_kws={"size": 5},
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.6, "label": "Interactions"}, ax=ax)
        ax.set_xlabel("Target"); ax.set_ylabel("Source")
        ax.set_title(f"{name}: Cell-Cell Interactions", fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=6); plt.yticks(fontsize=6)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/Fig4A_heatmap_{name}.pdf", bbox_inches="tight")
        plt.close()

    # 4B: Differential heatmap
    if "source" not in merged.columns or merged["source"].isna().all():
        merged["source"] = merged["interaction"].str.split(" -> ").str[0]
        merged["target"] = (merged["interaction"].str.split(" -> ").str[1]
                            .str.split(r" \| ").str[0])
    pair_diff = merged.groupby(["source", "target"])["diff"].mean().reset_index()
    pivot = pair_diff.pivot(index="source", columns="target", values="diff").fillna(0)
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 0.1)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, annot=False,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.6, "label": "Delta Rank (WT - KO)"}, ax=ax)
    ax.set_xlabel("Target"); ax.set_ylabel("Source")
    ax.set_title("Differential Interactions (WT vs KO)", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=6); plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/Fig4B_diff_heatmap.pdf", bbox_inches="tight"); plt.close()

    # 4C: Top differential L-R pairs
    stronger_ko = merged[merged["diff"] > 0.05].head(15)
    stronger_wt = merged[merged["diff"] < -0.05].head(15)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, subset, color, title in [
        (axes[0], stronger_ko, "#3C5488", "Stronger in KO"),
        (axes[1], stronger_wt, "#E64B35", "Stronger in WT"),
    ]:
        if len(subset) > 0:
            vals = subset["diff"].values if color == "#3C5488" else -subset["diff"].values
            ax.barh(range(len(subset)), vals, color=color, edgecolor="none", height=0.7)
            ax.set_yticks(range(len(subset)))
            labels = [
                f"{str(r['ligand_complex'])[:12]}:{str(r['receptor_complex'])[:12]}"
                for _, r in subset.iterrows()
            ]
            ax.set_yticklabels(labels, fontsize=5)
            ax.invert_yaxis()
        ax.set_title(title, fontweight="bold", color=color)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/Fig4C_top_diff_LR.pdf", bbox_inches="tight"); plt.close()

    # 4D: DC & Fibroblast focused
    for focus in ["DC", "Fibroblast"]:
        wt_f = wt_res[(wt_res["source"] == focus) | (wt_res["target"] == focus)].copy()
        ko_f = ko_res[(ko_res["source"] == focus) | (ko_res["target"] == focus)].copy()
        if len(wt_f) == 0 and len(ko_f) == 0:
            continue
        wt_f["lr"] = wt_f["ligand_complex"] + ":" + wt_f["receptor_complex"]
        ko_f["lr"] = ko_f["ligand_complex"] + ":" + ko_f["receptor_complex"]
        all_lr = list(set(wt_f["lr"].tolist() + ko_f["lr"].tolist()))
        plot_data = []
        for lr in all_lr:
            wt_sc = (1 - wt_f[wt_f["lr"] == lr]["magnitude_rank"].min()
                     if lr in wt_f["lr"].values else 0)
            ko_sc = (1 - ko_f[ko_f["lr"] == lr]["magnitude_rank"].min()
                     if lr in ko_f["lr"].values else 0)
            plot_data.append(dict(LR=lr, WT=wt_sc, KO=ko_sc, Diff=ko_sc - wt_sc))
        df = pd.DataFrame(plot_data).sort_values("Diff", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(5, max(3, len(df) * 0.25)))
        y = np.arange(len(df))
        h = 0.35
        ax.barh(y - h / 2, df["WT"], h, label="WT", color="#3C5488", edgecolor="none")
        ax.barh(y + h / 2, df["KO"], h, label="KO", color="#E64B35", edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels([lr[:25] for lr in df["LR"]], fontsize=5)
        ax.set_xlabel("Interaction Score (1 - rank)")
        ax.set_title(f"{focus} Communications", fontweight="bold")
        ax.legend(frameon=False, fontsize=6)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/Fig4D_{focus}_communications.pdf", bbox_inches="tight")
        plt.close()

    # 4E: Inflammation-related L-R
    inflam_kw = ["TNF", "IL1", "IL6", "IL12", "IL17", "IL23", "IFNG",
                 "CCL", "CXCL", "TGFB", "CD40", "CD80", "CD86", "MHC", "HLA"]

    def _inflam_mask(res):
        return (
            res["ligand_complex"].str.upper().apply(
                lambda x: any(k in str(x).upper() for k in inflam_kw)) |
            res["receptor_complex"].str.upper().apply(
                lambda x: any(k in str(x).upper() for k in inflam_kw))
        )

    wt_inf = wt_res[_inflam_mask(wt_res)].copy()
    ko_inf = ko_res[_inflam_mask(ko_res)].copy()

    if len(wt_inf) > 0 or len(ko_inf) > 0:
        wt_inf["lr"] = wt_inf["ligand_complex"] + ":" + wt_inf["receptor_complex"]
        ko_inf["lr"] = ko_inf["ligand_complex"] + ":" + ko_inf["receptor_complex"]
        all_lr = list(set(wt_inf["lr"].tolist() + ko_inf["lr"].tolist()))[:25]
        plot_data = []
        for lr in all_lr:
            wt_sc = (1 - wt_inf[wt_inf["lr"] == lr]["magnitude_rank"].min()
                     if lr in wt_inf["lr"].values else 0)
            ko_sc = (1 - ko_inf[ko_inf["lr"] == lr]["magnitude_rank"].min()
                     if lr in ko_inf["lr"].values else 0)
            plot_data.append(dict(LR=lr, WT=wt_sc, KO=ko_sc, Diff=ko_sc - wt_sc))
        df_inf = pd.DataFrame(plot_data).sort_values("Diff")
        colors = ["#E64B35" if d > 0 else "#3C5488" for d in df_inf["Diff"]]

        fig, ax = plt.subplots(figsize=(5, max(3, len(df_inf) * 0.22)))
        ax.barh(range(len(df_inf)), df_inf["Diff"], color=colors,
                edgecolor="none", height=0.7)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_yticks(range(len(df_inf)))
        ax.set_yticklabels([lr[:30] for lr in df_inf["LR"]], fontsize=5)
        ax.set_xlabel("Score Difference (KO - WT)")
        ax.set_title("Inflammation-related L-R Pairs", fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{fig_dir}/Fig4E_inflammation_LR.pdf", bbox_inches="tight")
        plt.close()
        df_inf.to_csv(f"{fig_dir}/inflammation_LR_comparison.csv", index=False)


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    adata = step1_qc_solo_scvi()
    adata = step2_umap_cluster(adata)
    adata = step3_celltypist(adata)
    adata = step4_scanvi(adata)

    fig1(adata)
    fig2(adata)
    fig3(adata)
    fig4(adata)

    print("Pipeline complete.")
