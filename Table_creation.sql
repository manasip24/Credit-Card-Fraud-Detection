CREATE TABLE IF NOT EXISTS transaction_details (
    transaction_id INT AUTO_INCREMENT PRIMARY KEY,
    amount DECIMAL(10, 2),
    transaction_time DATETIME,
    feature1 DECIMAL(10, 2),
    feature2 DECIMAL(10, 2),
    fraud_flag INT
);

CREATE TABLE IF NOT EXISTS credit_card_transactions (
    transaction_id INT AUTO_INCREMENT PRIMARY KEY,
    time INT,
    v1 DECIMAL(10, 2),
    v2 DECIMAL(10, 2),
    v3 DECIMAL(10, 2),
    v4 DECIMAL(10, 2),
    v5 DECIMAL(10, 2),
    v6 DECIMAL(10, 2),
    v7 DECIMAL(10, 2),
    v8 DECIMAL(10, 2),
    v9 DECIMAL(10, 2),
    v10 DECIMAL(10, 2),
    v11 DECIMAL(10, 2),
    v12 DECIMAL(10, 2),
    v13 DECIMAL(10, 2),
    v14 DECIMAL(10, 2),
    v15 DECIMAL(10, 2),
    v16 DECIMAL(10, 2),
    v17 DECIMAL(10, 2),
    v18 DECIMAL(10, 2),
    v19 DECIMAL(10, 2),
    v20 DECIMAL(10, 2),
    v21 DECIMAL(10, 2),
    v22 DECIMAL(10, 2),
    v23 DECIMAL(10, 2),
    v24 DECIMAL(10, 2),
    v25 DECIMAL(10, 2),
    v26 DECIMAL(10, 2),
    v27 DECIMAL(10, 2),
    v28 DECIMAL(10, 2),
    amount DECIMAL(10, 2),
    class INT
);

CREATE TABLE IF NOT EXISTS preprocessed_data (
    preprocessed_id INT AUTO_INCREMENT PRIMARY KEY,
    transaction_id INT,
    preprocessed_amount DECIMAL(10, 2),
    preprocessed_time DATETIME,
    preprocessed_feature1 DECIMAL(10, 2),
    preprocessed_feature2 DECIMAL(10, 2),
    preprocessed_fraud_flag INT
);

SELECT * FROM preprocessed_data LIMIT 5;


