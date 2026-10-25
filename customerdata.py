import datetime
import numpy as np
import pandas as pd

# Set seed for reproducible results
np.random.seed(42)

# Configuration Parameters
n_customers = 600  # Unique customers
n_transactions = 10000  # Total transaction rows
start_date = datetime.datetime(2023, 1, 1, 0, 0, 0)
end_date = datetime.datetime(2023, 12, 31, 23, 59, 59)

# Product Catalog across distinct Categories
catalog = [
    {
        "Category": "Electronics",
        "Items": [
            ("Wireless Mouse", 25.00),
            ("Mechanical Keyboard", 85.00),
            ("4K Monitor", 320.00),
            ("USB-C Cable", 12.00),
            ("Noise-Cancelling Headphones", 199.00),
        ],
    },
    {
        "Category": "Apparel",
        "Items": [
            ("Denim Jeans", 55.00),
            ("Cotton T-Shirt", 20.00),
            ("Hoodie", 45.00),
            ("Running Shoes", 90.00),
            ("Leather Belt", 30.00),
        ],
    },
    {
        "Category": "Home & Kitchen",
        "Items": [
            ("Coffee Maker", 75.00),
            ("Blender", 60.00),
            ("Stainless Steel Cookware", 150.00),
            ("Air Fryer", 110.00),
            ("Ceramic Mug Set", 22.00),
        ],
    },
    {
        "Category": "Beauty & Personal Care",
        "Items": [
            ("Skincare Serum", 35.00),
            ("Electric Toothbrush", 50.00),
            ("Hair Dryer", 65.00),
            ("Moisturizer", 28.00),
        ],
    },
    {
        "Category": "Books & Stationery",
        "Items": [
            ("Hardcover Notebook", 15.00),
            ("Fountain Pen", 40.00),
            ("Bestseller Fiction Novel", 18.00),
            ("Desk Organizer", 25.00),
        ],
    },
]

# Generate Customer Pool
customer_ids = [f"CUST_{1000 + i}" for i in range(n_customers)]

# Generate Transactions Data
rows = []
total_seconds = int((end_date - start_date).total_seconds())

for i in range(1, n_transactions + 1):
    # Group items into logical invoices (approx 2 items per invoice ID)
    invoice_no = f"INV_{100000 + (i // 2)}"

    # Non-uniform customer selection (simulates high/low frequency buyers)
    cust_id = np.random.choice(
        customer_ids, p=np.random.dirichlet(np.ones(n_customers))
    )

    # Random Category & Product Selection
    cat_info = np.random.choice(catalog)
    category_name = cat_info["Category"]
    product_name, unit_price = cat_info["Items"][
        np.random.randint(0, len(cat_info["Items"]))
    ]

    # Purchase Date & Time (random timestamp across the year)
    random_offset = np.random.randint(0, total_seconds)
    purchase_time = start_date + datetime.timedelta(seconds=random_offset)

    # Quantity & Total Spend Calculation
    quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.65, 0.20, 0.08, 0.04, 0.03])
    total_spend = round(quantity * unit_price, 2)

    rows.append([
        invoice_no,
        cust_id,
        purchase_time.strftime("%Y-%m-%d %H:%M:%S"),
        product_name,
        category_name,
        quantity,
        unit_price,
        total_spend,
    ])

# Build DataFrame
df = pd.DataFrame(
    rows,
    columns=[
        "InvoiceNo",
        "CustomerID",
        "PurchaseDateTime",
        "ProductDescription",
        "ProductCategory",
        "Quantity",
        "UnitPrice",
        "TotalSpend",
    ],
)

# Export to CSV
df.to_csv("customer_shopping_behavior.csv", index=False)
print(
    "Dataset generated successfully and saved to 'customer_shopping_behavior.csv'\n"
)
print(df.head(10))