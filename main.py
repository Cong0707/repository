import json
import multiprocessing
import os
import signal
import threading
from queue import Queue
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, request

# 步骤 1: 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# 模型架构
class BuildModel(nn.Module):
    def __init__(self, x_range, y_range, num_action_ids, num_building_ids):
        super(BuildModel, self).__init__()

        dim = 625

        # 更改卷积层的设置
        self.conv1d_1 = nn.Conv1d(5000, 64, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv1d_3 = nn.Conv1d(128, dim, kernel_size=3, padding=1)

        self.transformer = nn.Transformer(d_model=dim, nhead=5, num_encoder_layers=3)
        self.head_action_id = nn.Linear(dim, num_action_ids)
        self.head_x = nn.Linear(dim, x_range)
        self.head_y = nn.Linear(dim, y_range)
        self.head_building_id = nn.Linear(dim, num_building_ids)
        self.head_rotation = nn.Linear(dim, 4)

    def forward(self, input_data):
        input_data = torch.from_numpy(input_data).float().to(device)  # 将输入数据的数据类型更改为float
        # 使用多个卷积层进行特征映射
        output_conv = self.conv1d_1(input_data)
        output_conv = self.conv1d_2(output_conv)
        output_conv = self.conv1d_3(output_conv)
        # 将输出的特征维度调整为 (sequence_length, batch_size, d_model)
        output_conv = output_conv.permute(2, 0, 1)

        output = self.transformer(output_conv, output_conv)
        action_id = self.head_action_id(output).float()
        x = self.head_x(output).float()
        y = self.head_y(output).float()
        building_id = self.head_building_id(output).float()
        rotation = self.head_rotation(output).float()
        return action_id, x, y, building_id, rotation


def train_model(num_epochs, model, data, action_id, x, y, building_id, rotation):
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    # optimizer = optim.Adadelta(model.parameters(), rho=0.8)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        predicted_action_id, predicted_x, predicted_y, predicted_building_id, predicted_rotation = model(data)

        loss_action_id = criterion(predicted_action_id, action_id.to(device))
        loss_x = criterion(predicted_x, x.to(device))
        loss_y = criterion(predicted_y, y.to(device))
        loss_building_id = criterion(predicted_building_id, building_id.to(device))
        loss_rotation = criterion(predicted_rotation, rotation.to(device))

        total_loss = loss_action_id + loss_x + loss_y + loss_building_id + loss_rotation
        total_loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}")


#################################################################################################################

# 示例数据
# input_data = torch.randn(batch_size, sequence_length, input_size)

# 示例训练
# train_model(input_data, action_id, x, y, building_id)

# 示例推理
# predicted_action_id, predicted_x, predicted_y, predicted_building_id = predict_action(input_data)

#################################################################################################################

# 创建一个队列用于接收训练数据
data_queue = Queue()

# 创建一个标志来表示是否退出程序
exit_flag = multiprocessing.Event()

print("loading model")
if os.path.exists("model_info.pth"):
    print("loading old model")
    model_info = torch.load('model_info.pth')
    # 构建模型
    model = BuildModel(model_info['x_range'], model_info['y_range'], model_info['num_action_ids'],
                       model_info['num_building_ids'])
    # 加载模型的权重
    model.load_state_dict(torch.load('model_weights.pth'))
else:
    print("making new models")
    model_info = {
        'x_range': 5000,
        'y_range': 5000,
        'num_action_ids': 3,
        'num_building_ids': 254
    }
    model = BuildModel(model_info['x_range'], model_info['y_range'], model_info['num_action_ids'],
                       model_info['num_building_ids'])


# 步骤 5: 启动训练循环的函数
def start_training():
    global model
    # 启动训练循环
    print("start training process")
    num_epochs = 5
    while not exit_flag.is_set():
        if data_queue.empty():
            continue
        data, action_id, x, y, building_id, rotation = data_queue.get(timeout=0.000001)
        print("learning")
        model = model.to(device)  # 将模型移动到合适的设备上
        model.train()
        train_model(num_epochs, model, data, action_id, x, y, building_id, rotation)
        print("Training finished.")
        torch.cuda.empty_cache()

    # 在程序退出时保存模型
    if model is not None:
        # 保存模型的权重和结构
        torch.save(model.state_dict(), 'model_weights.pth')
        # 保存模型的其他相关信息（例如，模型架构和超参数）
        torch.save(model_info, 'model_info.pth')
        print("Model saved.")


# 信号处理函数，用于捕获 Ctrl+C 信号
def signal_handler(signum, frame):
    print("Ctrl+C detected. Stopping training and exiting gracefully.")
    exit_flag.set()
    exit(0)


# 注册信号处理函数，以捕获 Ctrl+C 信号
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # 启动训练循环的线程
    training_thread = threading.Thread(target=start_training)
    training_thread.start()

    # 创建一个 Flask 应用
    app = Flask(__name__)

    # 定义一个简单的路由，接收 POST 请求并将数据放入队列
    @app.route('/api/ai', methods=['POST'])
    def ai():
        data_bytes = request.data
        postData = data_bytes.decode('utf-8')
        data = json.loads(postData)

        # 获取x和y的最大值，用于初始化二维数组
        # num_rows = max(int(item['x']) for item in data[1])
        # num_cols = max(int(item['y']) for item in data[1])

        # 初始化三维数组
        world = numpy.full((5000, 5000, 5), 0)

        # 将数据填充到数组中
        for item in data[1]:
            x, y = int(item['x']), int(item['y'])
            world[x][y][0] = int(item['floor'])
            world[x][y][1] = int(item['ore'])
            world[x][y][2] = int(item['block'])
            world[x][y][3] = int(item['face'])
            world[x][y][4] = int(item['overlay'])

        action_id = torch.tensor(int(data[2]['action'])).float()  # 将action_id转换为张量
        x = torch.tensor(int(data[2]['x'])).float()  # 将x坐标转换为张量
        y = torch.tensor(int(data[2]['y'])).float()  # 将y坐标转换为张量
        building_id = torch.tensor(int(data[2]['id'])).float()  # 将building_id转换为张量
        rotation = torch.tensor(int(data[2]['rotation'])).float()  # 将rotation转换为张量

        data_queue.put((world, action_id, x, y, building_id, rotation))

        return jsonify(message="Data received and queued for training.")


    @app.route('/api/build', methods=['POST'])
    def build():
        global model
        data_bytes = request.data
        postData = data_bytes.decode('utf-8')
        data = json.loads(postData)

        # 初始化三维数组
        world = numpy.full((5000, 5000, 5), 0)

        # 将数据填充到数组中
        for item in data:
            x, y = int(item['x']), int(item['y'])
            world[x][y][0] = int(item['floor'])
            world[x][y][1] = int(item['ore'])
            world[x][y][2] = int(item['block'])
            world[x][y][3] = int(item['face'])
            world[x][y][4] = int(item['overlay'])

        model.eval()  # 设置为评估模式
        with torch.no_grad():
            # 将数据移动到合适的设备上
            action_id, x, y, building_id, rotation = model(world)
        return jsonify(action=action_id, x=x, y=y, building=building_id, rotation=rotation)


    try:
        # 运行 Flask 应用
        app.run(debug=False)
    except KeyboardInterrupt:
        print("Ctrl+C detected. Stopping Flask server.")

    training_thread.join()
