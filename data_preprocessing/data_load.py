import pandas as pd
from datetime import datetime
import os


def load_and_clean_data(file_path):
    # 加载原始日志数据
    df = pd.read_csv(file_path)
    
    # 清洗数据：去除空值，统一活动名称格式
    df = df.dropna(subset=['Case ID', 'Activity', 'Resource', 'Complete Timestamp'])
    df = df.rename(columns={
        'Case ID': 'case_id',
        'Activity': 'activity',
        'Resource': 'resource',
        'Complete Timestamp': 'timestamp'
    })
    df['activity'] = df['activity'].str.strip().str.lower()
    
    # 解析时间戳（假设列为 'timestamp'）
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['case_id', 'activity', 'resource', 'timestamp']]
    # 按案例ID和时间排序
    df = df.sort_values(['case_id', 'timestamp'])
    
    output_dir = '../data'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_processed.csv'))
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    # 示例：加载数据
    print(f"Current working directory: {os.getcwd()}")
    file_path = '../data/helpdesk.csv'
    cleaned_data = load_and_clean_data(file_path)
    
    # 打印清洗后的数据
    print(cleaned_data.head())