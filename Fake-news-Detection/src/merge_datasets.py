import pandas as pd

def merge_true_fake(true_path='data/True.csv', fake_path='data/Fake.csv', output_path='data/fake_or_real_news.csv'):
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    # Make sure these dataframes are valid
    true_df['label'] = 1  # REAL
    fake_df['label'] = 0  # FAKE

    # Important: verify text column exists
    if 'text' not in true_df.columns or 'text' not in fake_df.columns:
        raise ValueError("Missing 'text' column in True.csv or Fake.csv")

    # Drop any rows where text is null
    true_df = true_df.dropna(subset=['text'])
    fake_df = fake_df.dropna(subset=['text'])

    merged_df = pd.concat([true_df, fake_df], axis=0)
    merged_df = merged_df[['title', 'text', 'label']]
    merged_df = merged_df.dropna()  # drop any row with NaN
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    merged_df.to_csv(output_path, index=False)
    print(f"✅ Merged dataset saved to: {output_path}")
