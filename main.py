import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Set plot style for clean VS Code figures
sns.set_theme(style="whitegrid")


def run_customer_segmentation():
    # ==========================================
    # Task 1: Load the Data
    # ==========================================
    csv_file = "customer_shopping_behavior.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"'{csv_file}' not found in current directory! Please check file path."
        )

    print("--- 1. Loading & Inspecting Raw Data ---")
    df = pd.read_csv(csv_file)
    print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nData Schema Overview:")
    print(df.info())
    print("\nMissing Values Check:\n", df.isnull().sum())

    # ==========================================
    # Task 2: Data Pre-processing & Feature Engineering
    # ==========================================
    print("\n--- 2. Pre-processing & Feature Engineering ---")

    # Clean duplicates & format timestamps
    df.drop_duplicates(inplace=True)
    df["PurchaseDateTime"] = pd.to_datetime(df["PurchaseDateTime"])

    # Reference date for recency calculation (1 day past the latest purchase)
    snapshot_date = df["PurchaseDateTime"].max() + pd.Timedelta(days=1)

    # Aggregate transaction-level data to customer level
    customer_df = (
        df.groupby("CustomerID")
        .agg(
            Recency=(
                "PurchaseDateTime",
                lambda x: (snapshot_date - x.max()).days,
            ),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalSpend", "sum"),
            TotalQuantity=("Quantity", "sum"),
            CategoryDiversity=("ProductCategory", "nunique"),
        )
        .reset_index()
    )

    # Calculate average basket size per order
    customer_df["AverageBasketSize"] = (
        customer_df["Monetary"] / customer_df["Frequency"]
    )

    print(
        f"Aggregated transaction log into {len(customer_df)} unique customer profiles."
    )
    print(customer_df.head())

    # ==========================================
    # Task 3 & 5: Feature Selection & Standardization
    # ==========================================
    print("\n--- 3. Feature Standardization ---")
    feature_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "AverageBasketSize",
        "CategoryDiversity",
    ]
    X = customer_df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==========================================
    # Task 6 & 7: Model Training & Performance Evaluation
    # ==========================================
    print("\n--- 4. Evaluating K-Means Cluster Metrics ---")
    inertia = []
    silhouette_scores = []
    K_range = range(2, 9)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"Clusters K={k} | Silhouette Score: {score:.4f}")

    # Plot 1: Elbow Curve & Silhouette Scores
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia (Elbow)", color="tab:blue", fontsize=12)
    ax1.plot(K_range, inertia, "o-", color="tab:blue", linewidth=2)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="tab:red", fontsize=12)
    ax2.plot(K_range, silhouette_scores, "s--", color="tab:red", linewidth=2)
    plt.title(
        "Elbow Curve & Silhouette Score Analysis",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("elbow_silhouette_analysis.png")
    plt.show()

    # Train Final KMeans Model (Optimal K = 4)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    customer_df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Hierarchical Agglomerative Clustering Comparison
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
    customer_df["Agg_Cluster"] = agg_clustering.fit_predict(X_scaled)

    # PCA 2D Dimensionality Reduction
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(X_scaled)
    customer_df["PCA1"] = pca_transformed[:, 0]
    customer_df["PCA2"] = pca_transformed[:, 1]

    # Plot 2: 2D PCA Cluster Plot
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=customer_df,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette="tab10",
        s=80,
        alpha=0.85,
    )
    plt.title(
        "Customer Loyalty Segments Visualized via PCA",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} Var)")
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} Var)")
    plt.tight_layout()
    plt.savefig("pca_customer_clusters.png")
    plt.show()

    # Summary Statistics & Profiling Table
    print("\n==========================================")
    print("      FINAL CLUSTER PROFILE TABLE")
    print("==========================================")
    print("\nCluster Distribution:")
    print(customer_df["Cluster"].value_counts())

    print("\nMean Feature Values per Segment:")
    profile_summary = customer_df.groupby("Cluster")[feature_cols].mean()
    print(profile_summary.round(2))

    # Save Results
    customer_df.to_csv("segmented_customer_output.csv", index=False)
    print(
        "\nExecution completed! Segmented dataset saved to 'segmented_customer_output.csv'."
    )


if __name__ == "__main__":
    run_customer_segmentation()



    import os
import pandas as pd

# Define input files
SEGMENTED_FILE = "segmented_customer_output.csv"
TRANSACTIONS_FILE = "customer_shopping_behavior.csv"


def load_data():
    if not os.path.exists(SEGMENTED_FILE) or not os.path.exists(
        TRANSACTIONS_FILE
    ):
        print("Error: Missing required CSV files.")
        print(
            f"Please ensure both '{SEGMENTED_FILE}' and '{TRANSACTIONS_FILE}' exist in this folder."
        )
        return None, None

    segmented_df = pd.read_csv(SEGMENTED_FILE)
    transactions_df = pd.read_csv(TRANSACTIONS_FILE)
    return segmented_df, transactions_df


def lookup_customer():
    segmented_df, transactions_df = load_data()
    if segmented_df is None:
        return

    # Persona mapping for clusters
    segment_names = {
        0: "Potential Loyalists",
        1: "Loyal Customers",
        2: "At-Risk / Hibernating",
        3: "VIP Champions",
    }

    print("==================================================")
    print("      CUSTOMER CLUSTER & HISTORY LOOKUP TOOL      ")
    print("==================================================")
    print("Type a CustomerID (e.g., CUST_1001) or 'exit' to quit.\n")

    while True:
        user_input = input("\nEnter Customer ID: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting lookup tool. Goodbye!")
            break

        # Search customer in segmented dataset
        cust_profile = segmented_df[
            segmented_df["CustomerID"].str.upper() == user_input.upper()
        ]

        if cust_profile.empty:
            print(
                f"\n❌ Customer ID '{user_input}' not found! Please check the ID and try again."
            )
            continue

        cust_data = cust_profile.iloc[0]
        cluster_id = int(cust_data["Cluster"])
        persona = segment_names.get(cluster_id, "Unknown Segment")

        print("\n" + "=" * 50)
        print(f"       PROFILE FOR: {cust_data['CustomerID']}")
        print("=" * 50)
        print(f"📌 Assigned Cluster : Cluster {cluster_id} ({persona})")
        print(f"⏱️  Recency          : {cust_data['Recency']} days ago")
        print(f"🛍️  Frequency        : {cust_data['Frequency']} total orders")
        print(f"💰 Total Spend       : ${cust_data['Monetary']:,.2f}")
        print(
            f"📦 Avg Basket Size   : ${cust_data['AverageBasketSize']:,.2f} per order"
        )
        print(
            f"🏷️  Category Diversity: {cust_data['CategoryDiversity']} unique categories"
        )

        # Retrieve full purchase history from raw transactions
        history = transactions_df[
            transactions_df["CustomerID"].str.upper() == user_input.upper()
        ].sort_values("PurchaseDateTime", ascending=False)

        print("\n--- Transaction History (Most Recent First) ---")
        history_display = history[[
            "InvoiceNo",
            "PurchaseDateTime",
            "ProductCategory",
            "ProductDescription",
            "Quantity",
            "UnitPrice",
            "TotalSpend",
        ]]
        print(history_display.to_string(index=False))
        print("=" * 50)


if __name__ == "__main__":
    lookup_customer()