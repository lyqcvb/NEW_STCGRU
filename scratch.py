import numpy as np
import scipy.stats as stats
import pandas as pd
import nolds
from scipy.signal import welch
import mne 
import logging
import os
import warnings
from scipy.stats import entropy
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
# 时域特征计算
def calculate_time_domain_features(data):
    features = {}
    features['mean'] = np.mean(data, axis=1)
    features['std'] = np.std(data, axis=1)
    features['skew'] = stats.skew(data, axis=1)
    features['kurtosis'] = stats.kurtosis(data, axis=1)

    time_df = pd.DataFrame(features)
    time_df['Channel'] = np.arange(1, 20) 
    time_df = time_df.set_index('Channel')
    return time_df

# 频域特征计算
def calculate_psd_features(data, fs=500):
    psd_all_channels = []
    n_channels = data.shape[0]
    
    # 计算每个通道的 PSD
    for i in range(n_channels):
        f, Pxx = welch(data[i, :], fs=fs, nperseg=512)  # 使用Welch方法计算功率谱密度
        
        # 找到频率在 0.5 到 32 Hz 之间的索引
        freq_range = (f >= 0.5) & (f <= 32)
        # 提取相应频率和功率谱密度值
        Pxx_filtered = Pxx[freq_range]
        psd_all_channels.append(Pxx_filtered)

    # 创建 DataFrame
    psd_df = pd.DataFrame(psd_all_channels)
    
    # 为列名生成频率标签
    freq_labels = [f'{freq:.1f}_hz' for freq in f[freq_range]]
    psd_df.columns = freq_labels
    
    # 添加通道列并设置为索引
    psd_df['Channel'] = np.arange(1, n_channels + 1)
    psd_df = psd_df.set_index('Channel')

    return psd_df

def calculate_permutation_entropy(data, order=3, delay=1):

    pe_all_channels = []
    n_channels = data.shape[0]
    
    # 计算每个通道的 PE
    for i in range(n_channels):
        # Step 3: Extract subsequences and sort them
        n = len(data[i])
        time_series = data[i]
        permutations = np.array([np.argsort(np.take(time_series, list(range(j, j + order * delay, delay)), mode='wrap')) for j in range(n - (order - 1) * delay)])
        # Step 4: Count occurrences of each permutation
        counts = np.array([len(np.where(permutations == p)[0]) for p in set(tuple(x) for x in permutations)])
        # Normalize to get probabilities
        probabilities = counts / float(sum(counts))
        # Step 5: Calculate entropy
        pe = entropy(probabilities)
        pe_all_channels.append(pe)
    pe_df = pd.DataFrame(pe_all_channels)
    pe_df['Channel'] = np.arange(1, 20) 
    pe_df = pe_df.set_index('Channel')
    return pe_df

def save_feature(file_paths,type,base_dir):
    for i, file in enumerate(file_paths):
        # try:
            # raw = mne.io.read_raw_edf(file, preload=True, encoding='latin1',verbose='Warning')
            # data = raw.get_data()[0:19]
            data = np.array(pd.read_csv(file))[:,1:]
            # 计算时域特征
            feature = calculate_psd_features(data)
            file_name = file[16:19]
            name = file[20:]
            if file_name[-1]=='\\':
                file_name=file_name[0:len(file_name)-1]
                name = file[19:]
            if(type == 0):
                dir = base_dir+"\\PSD\\认知正常\\"+file_name+'\\'
                ensure_dir(dir)
                file_path = dir+name
                feature.to_csv(file_path)
            else:
                dir = base_dir+"\\PSD\\认知障碍\\"+file_name+'\\'
                ensure_dir(dir)
                file_path = dir+name
                feature.to_csv(file_path)
            print(i , " has been processed")
        # except Exception as e:
        #     logging.error(f"Error processing file {file}: {e}")
        #     continue

# 忽略 RuntimeWarning 警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
# 定义文件夹路径
base_dir = '糖尿病数据ICA分段'
normal_dir = os.path.join(base_dir, '认知正常')
impaired_dir = os.path.join(base_dir, '认知障碍')

# 获取所有的文件路径


normal_files_names = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) ]
normal_files = [os.path.join(x, f) for x in  normal_files_names for f in os.listdir(x ) if f.endswith('.csv')]
impaired_files_names = [os.path.join(impaired_dir, f) for f in os.listdir(impaired_dir)]
impaired_files = [os.path.join(x, f) for x in  impaired_files_names for f in os.listdir(x ) if f.endswith('.csv')]
save_dir = "糖尿病数据特征"
save_feature(normal_files,0,save_dir)
save_feature(impaired_files,1,save_dir)