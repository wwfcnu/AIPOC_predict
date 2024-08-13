import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from torch.utils.tensorboard import SummaryWriter
import random
import time
# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 你可以选择任何一个数字作为随机种子

# 加载CSV文件
train_file_path = '/mnt/data/laion-ai/AIPOC/train_file_1_30.csv'
val_file_path = '/mnt/data/laion-ai/AIPOC/test_file31.csv'
train_data = pd.read_csv(train_file_path)
val_data = pd.read_csv(val_file_path)

# 初始化 MinMaxScaler
scaler = MinMaxScaler()
train_data[['HeatCoeff', 'T0']] = scaler.fit_transform(train_data[['HeatCoeff', 'T0']])
val_data[['HeatCoeff', 'T0']] = scaler.transform(val_data[['HeatCoeff', 'T0']])


# 数据预处理函数
def process_data(data):
    # 提取三维坐标点和温度
    x = data['X'].values
    y = data['Y'].values
    z = data['Z'].values
    temperature = data['Temp'].values
    time = data['Time'].values
    heat_coeff = data['HeatCoeff'].values
    T0 = data['T0'].values

    # 转换为柱坐标
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    return x, y, r, z, temperature, time, heat_coeff, T0

def create_grid(r, z, temperature, time, heat_coeff, T0):
    # Extract unique values for r and z
    unique_r = np.unique(r)
    unique_z = np.unique(z)

    # Create the mesh grid using the unique values
    r_grid, z_grid = np.meshgrid(unique_r, unique_z, indexing='ij')

    # Interpolate the data onto the unique grid using cubic interpolation
    temperature_grid = griddata((r, z), temperature, (r_grid, z_grid), method='cubic')
    time_grid = griddata((r, z), time, (r_grid, z_grid), method='cubic')
    heat_coeff_grid = griddata((r, z), heat_coeff, (r_grid, z_grid), method='cubic')
    T0_grid = griddata((r, z), T0, (r_grid, z_grid), method='cubic')

    # Fill missing values using nearest-neighbor interpolation for NaNs that might still exist
    temperature_grid = griddata((r, z), temperature, (r_grid, z_grid), method='nearest')
    time_grid = griddata((r, z), time, (r_grid, z_grid), method='nearest')
    heat_coeff_grid = griddata((r, z), heat_coeff, (r_grid, z_grid), method='nearest')
    T0_grid = griddata((r, z), T0, (r_grid, z_grid), method='nearest')

    return r_grid, z_grid, temperature_grid, time_grid, heat_coeff_grid, T0_grid



# 提取数据中的独特工况和时间点组合
def extract_conditions(data):
    return data.groupby(['Time', 'HeatCoeff', 'T0']).size().reset_index().iloc[:, :3]

# train_conditions_times = extract_conditions(train_data)
val_conditions_times = extract_conditions(val_data)

# 创建所有工况和时间点的网格

# 初始化存储所有二维网格的列表
def create_all_grids(data, conditions_times):
    all_grids = []
    all_temperatures = []

    all_data = []
    all_grid_data = []


    # 遍历每个工况和时间点组合
    for _, row in conditions_times.iterrows():
        condition_data = data[(data['Time'] == row['Time']) & (data['HeatCoeff'] == row['HeatCoeff']) & (data['T0'] == row['T0'])]
        x, y, r, z, temperature, time, heat_coeff, T0 = process_data(condition_data)
        # 保存为csv，表头为X，Y，Z，r，Temp，Time，HeatCoeff,T0

        # 保存原始数据
        data_dict = {
            'X': x,
            'Y': y,
            'Z': z,
            'r': r,
            'Temp': temperature,
            'Time': time,
            'HeatCoeff': heat_coeff,
            'T0': T0
        }
        all_data.append(pd.DataFrame(data_dict))

       


        r_grid, z_grid, temperature_grid, time_grid, heat_coeff_grid, T0_grid = create_grid(r, z, temperature, time, heat_coeff, T0)
        all_grids.append(np.stack([r_grid, z_grid, time_grid, heat_coeff_grid, T0_grid], axis=0))
        all_temperatures.append(temperature_grid)



        #将网格数据展开并合并成DataFrame
        grid_data = {
            'r': r_grid.flatten(),
            'Z': z_grid.flatten(),
            'Time': time_grid.flatten(),
            'HeatCoeff': heat_coeff_grid.flatten(),
            'T0': T0_grid.flatten(),
            'Temp': temperature_grid.flatten()
        }

        all_grid_data.append(pd.DataFrame(grid_data))

    # 合并所有DataFrame
    all_data_df = pd.concat(all_data, ignore_index=True)
    all_grid_data_df = pd.concat(all_grid_data, ignore_index=True)
    return np.array(all_grids), np.array(all_temperatures), all_data_df, all_grid_data_df




# all_train_grids, all_train_temperatures = create_all_grids(train_data, train_conditions_times)
# all_val_grids, all_val_temperatures = create_all_grids(val_data, val_conditions_times)

all_val_grids, all_val_temperatures ,all_val_data_df, all_val_grid_data_df= create_all_grids(val_data, val_conditions_times)


# print(f"Total 2D grids for training: {all_train_grids.shape[0]}")
print(f"Total 2D grids for validation: {all_val_grids.shape}")

# 定义自定义数据集
class TemperatureDataset(Dataset):
    def __init__(self, grids, temperatures):
        self.grids = torch.tensor(grids, dtype=torch.float32)
        self.temperatures = torch.tensor(temperatures, dtype=torch.float32)

    def __len__(self):
        return self.grids.shape[0]

    def __getitem__(self, idx):
        return self.grids[idx], self.temperatures[idx]

# 创建数据集和数据加载器
# train_dataset = TemperatureDataset(all_train_grids, all_train_temperatures)
val_dataset = TemperatureDataset(all_val_grids, all_val_temperatures)

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)   #修改batch_size为8
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 打印部分train_dataset中的样本
# for i in range(5):  # 打印前5个样本
#     grid, temperature = train_dataset[i]
#     print(f"Sample {i + 1}:")
#     print("Grid shape:", grid.shape)
#     print("Grid data:", grid)
#     print("Temperature shape:", temperature.shape)
#     print("Temperature data:", temperature)
#     print("\n")




# 定义CNN模型
class TemperatureCNN(nn.Module):
    def __init__(self):
        super(TemperatureCNN, self).__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*60*10, 128) 
        self.fc2 = nn.Linear(128, 242*41)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), 242, 41)
        return x

# 创建模型、定义损失函数和优化器
model = TemperatureCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# 训练函数
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    return avg_loss

# 验证函数
# def validate_one_epoch(model, dataloader, criterion):
#     model.eval()
#     running_loss = 0.0
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             running_loss += loss.item()
#     avg_loss = running_loss / len(dataloader)
#     return avg_loss
def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    avg_loss = running_loss / len(dataloader)

    all_labels = np.concatenate(all_labels).reshape(-1, 9922)
    all_outputs = np.concatenate(all_outputs).reshape(-1, 9922)

    mse = mean_squared_error(all_labels, all_outputs)
    rmse = np.sqrt(mean_squared_error(all_labels, all_outputs))
    mae = mean_absolute_error(all_labels, all_outputs)
    r2 = r2_score(all_labels, all_outputs)

    return avg_loss,mse, rmse, mae, r2
# # 训练模型
# num_epochs = 500

# # Initialize TensorBoard writer
# writer = SummaryWriter(log_dir='./runs/temperature_regression')




# for epoch in range(num_epochs):
#     train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
#     val_loss, mse, rmse, mae, r2= validate_one_epoch(model, val_loader, criterion)
#     #scheduler.step()

#     print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')
#     print(f'val MSE:{mse}, RMSE:{rmse}, MAE:{mae}, R2:{r2}')


#     # Log the losses to TensorBoard
#     writer.add_scalar('Loss/Train', train_loss, epoch)
#     writer.add_scalar('Loss/Eval', val_loss, epoch)


# # Close the TensorBoard writer
# writer.close()

# # 保存模型
# torch.save(model.state_dict(), "/mnt/data/laion-ai/AIPOC/cnn_model_weights.pth")





# 加载模型
start = time.time()
model = TemperatureCNN()
model.load_state_dict(torch.load("/mnt/data/laion-ai/AIPOC/cnn_model_weights.pth"))


# 进行预测
# def predict(model, dataloader):
#     model.eval()
#     predictions = []
#     all_inputs = []
#     all_labels = []

#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             outputs = model(inputs)
            
#             # 将inputs, labels, outputs存储起来
#             all_inputs.append(inputs.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())
#             predictions.append(outputs.cpu().numpy())

#     return np.array(all_inputs), np.array(all_labels), np.array(predictions)

# 进行预测并保存数据
def predict_and_save(model, dataloader,  data_df, original_grid_data_df, filename='predicted_data.csv'):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0).reshape(-1)
    original_grid_data_df['predict_temperature'] = predictions

   
    # 将data_df左连接original_grid_data_df，按照r, Z, Temp, Time, HeatCoeff, T0
    merged_df = pd.merge(data_df, original_grid_data_df, on=['r', 'Z', 'Temp', 'Time', 'HeatCoeff', 'T0'], how='left')
    merged_df.to_csv(filename, index=False)
    print(f"数据已成功保存到 {filename}")




# 调用预测函数并保存数据
predict_and_save(model, val_loader,all_val_data_df, all_val_grid_data_df)
