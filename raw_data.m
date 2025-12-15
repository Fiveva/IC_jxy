%% 神经元活动强度波形图 (集成 Delta F/F 归一化)

mouse_id = 21; % <-- 设为 21 处理 M21 数据，设为 79 处理 M79 数据

%% --- 1. 文件和参数定义 ---
% 动态构建文件名
MAT_FILE = sprintf('%d_wholebrain_output.mat', mouse_id);
% 注意：如果 TXT 文件名中包含反引号，需要确保 MATLAB 正确处理路径。
TXT_FILE = sprintf('IC\\indexed_mod_2025`024_M%d.txt', mouse_id);
TITLE_TEXT = sprintf('M%d 神经元活动强度波形图 (Delta F/F)', mouse_id); % 修改标题以反映DFF

% 试次 (Trial) 窗口参数
PRE_EVENT_FRAMES = 12; % 刺激前帧数 (行)
POST_EVENT_FRAMES = 20; % 刺激后帧数 (行)

% 定义变量
TOTAL_TRIAL_FRAMES = PRE_EVENT_FRAMES + POST_EVENT_FRAMES + 1; % 总共33个数据点 (12+1+20)
CENTER_FRAME_INDEX = PRE_EVENT_FRAMES + 1; % 刺激发生时刻在窗口中的位置

TOTAL_LINES_TO_CHECK = PRE_EVENT_FRAMES + POST_EVENT_FRAMES + 1; % 33
TOTAL_DATA_POINTS = PRE_EVENT_FRAMES + POST_EVENT_FRAMES; % 最终绘图点数：32

fprintf('--- 神经元活动强度分析开始 (M%d) ---\n', mouse_id);

%% --- 2. 加载数据并进行 Delta F/F 归一化 ---
try
    % 1. 加载钙成像轨迹数据
    data_struct = load(MAT_FILE, 'whole_trace_ori');
    whole_trace_ori = data_struct.whole_trace_ori;
    [num_neurons, num_frames] = size(whole_trace_ori);
    fprintf('成功加载MAT文件。神经元数量: %d, 拍照记录数 (帧): %d\n', num_neurons, num_frames);
    
    
    % --- 新增：Delta F / F 归一化处理 ---
    
    % 定义基线窗口大小 (需要比最长钙信号衰减时间更长)
    BASELINE_WINDOW_FRAMES = 300; 
    
    whole_trace_dff = zeros(num_neurons, num_frames, 'single');
    
    fprintf('开始进行 Delta F / F 归一化 (F0 窗口: %d 帧)...\n', BASELINE_WINDOW_FRAMES);
    
    % 检查是否可以使用 movmin 函数 (需要 Signal Processing Toolbox 或较新版本 MATLAB)
    if exist('movmin', 'builtin') ~= 5
        warning('MATLAB:MissingFunction', 'movmin 函数不可用，跳过 DFF 归一化。请确保安装了 Signal Processing Toolbox。');
        % 如果 movmin 不可用，则保留原始数据
        whole_trace_ori_dff = whole_trace_ori;
    else
        
        for i = 1:num_neurons
            F = whole_trace_ori(i, :);
            
            % 1. 估计基线 F0
            % 使用滚动最小值 (movmin) 作为基线 F0 的近似。
            % 'Endpoints', 'fill' 确保输出长度与输入相同。
            F0 = movmin(F, BASELINE_WINDOW_FRAMES, 'omitnan', 'Endpoints', 'fill'); 
            
            % 2. 确保 F0 永远大于一个很小的正数，防止除以零
            F0(F0 <= 1e-6) = 1e-6; 
            
            % 3. 计算 Delta F / F = (F - F0) / F0
            dff = (F - F0) ./ F0;
            whole_trace_dff(i, :) = dff;
        end
        
        % 替换原始变量，后续分析使用归一化后的数据
        whole_trace_ori = whole_trace_dff;
        fprintf('Delta F / F 归一化完成，数据已更新。\n');
    end
    % --- Delta F / F 转换结束 ---


catch
    error('错误：无法加载 %s 文件或其中不包含变量 whole_trace_ori。', MAT_FILE);
end

% 2. 读取文本文件内容
try
    % 使用 readlines 读取文件所有行到字符串数组
    event_lines = readlines(TXT_FILE);
    num_lines = length(event_lines);
    fprintf('成功读取文本文件。总行数: %d\n', num_lines);
catch
    error('错误：无法读取 %s 文件。', TXT_FILE);
end

%% --- 3. 解析文本文件，提取刺激时间点 (Trial 索引) ---
% 初始化映射数组：存储每一行对应的拍照帧索引。如果该行不是拍照点，则为 NaN。
% 大小与文本行数相同
line_to_frame_map = NaN(num_lines, 1);
stimulus_line_indices = []; % 存储所有 ' a' 刺激开始行的行号

fprintf('开始解析文本文件，建立行号到帧号映射...\n');

for line_idx = 1:num_lines
    line = event_lines(line_idx);
    
    % a. 识别刺激开始行 (以 " a" 结尾)
    if endsWith(line, ' a')
        stimulus_line_indices = [stimulus_line_indices; line_idx];
        % 刺激行本身没有拍照帧号，map中保持为 NaN
        
    % b. 识别拍照点行 (以 " " + 数字索引结尾)
    else
        % 尝试提取行末尾的数字索引
        tokens = regexp(line, '\s+(\d+)$', 'tokens', 'once');
        
        if ~isempty(tokens)
            frame_index = str2double(tokens{1});
            
            % 检查索引是否有效
            if ~isnan(frame_index) && frame_index >= 1 && frame_index <= num_frames
                line_to_frame_map(line_idx) = frame_index;
            else
                % 找到数字但无效的情况 (可忽略警告)
                % fprintf('警告：行 %d 提取到无效帧索引: %s\n', line_idx, tokens{1});
            end
        end
    end
end

num_trials = length(stimulus_line_indices);
fprintf('共找到 %d 个 Trial 刺激开始行。\n', num_trials);

if num_trials == 0
    error('未找到任何有效的刺激开始点，程序终止。');
end

%% --- 4. 提取和平均 Trial 数据 ---

% 预分配矩阵来存储所有 Trial 的有效平均轨迹 (32个点)
all_trial_traces = NaN(num_trials, TOTAL_DATA_POINTS); 
valid_trials_count = 0; % 统计有效Trial的数量

fprintf('开始提取和平均 Trial 数据...\n');

% 遍历每个刺激开始行的行号
for t = 1:num_trials
    stim_line_idx = stimulus_line_indices(t); % 刺激开始的行号
    
    % 计算 Trial 窗口的文本行号范围
    start_line = stim_line_idx - PRE_EVENT_FRAMES;
    end_line = stim_line_idx + POST_EVENT_FRAMES;
    
    % 确保 Trial 窗口在文本文件行号范围内
    if start_line >= 1 && end_line <= num_lines
        
        % (1) 提取该 Trial 窗口内所有行的拍照帧索引
        trial_line_indices = start_line:end_line;
        trial_frame_indices_all = line_to_frame_map(trial_line_indices);
        
        % (2) 移除刺激开始行对应的 NaN (它位于第 PRE_EVENT_FRAMES + 1 个位置)
        % 得到 32 个有效或潜在有效的帧索引
        trial_frame_indices_32 = [trial_frame_indices_all(1:PRE_EVENT_FRAMES); 
                                  trial_frame_indices_all(PRE_EVENT_FRAMES+2:end)];
        
        % (3) 仅保留有效的帧索引，并记录它们在 32 个点中的位置
        valid_frame_mask = ~isnan(trial_frame_indices_32);
        valid_frame_indices = trial_frame_indices_32(valid_frame_mask);
        
        if isempty(valid_frame_indices)
             fprintf('警告：Trial %d (刺激行 %d) 窗口内没有找到任何有效的拍照帧索引，跳过。\n', t, stim_line_idx);
             continue;
        end
        
        % (4) 从 whole_trace_ori 中取出对应列数据 (此时 whole_trace_ori 已是 DFF 数据)
        trial_data = whole_trace_ori(:, valid_frame_indices);
        
        % (5) 每一列取所有神经元的平均
        trial_average_trace_valid = mean(trial_data, 1, 'omitnan');
        
        % (6) 将平均轨迹放回 32 个点的向量中对应的位置 (无效帧的位置仍为 NaN)
        trial_average_trace_32 = NaN(1, TOTAL_DATA_POINTS);
        trial_average_trace_32(valid_frame_mask) = trial_average_trace_valid;
        
        % (7) 记录这 32 个数据
        all_trial_traces(t, :) = trial_average_trace_32;
        valid_trials_count = valid_trials_count + 1; % 有效Trial计数
        
    else
        % 如果 Trial 窗口超出范围
        fprintf('警告：Trial %d (刺激行 %d) 窗口超出文本文件范围，跳过。\n', t, stim_line_idx);
    end
end

% (8) 将所有有效 Trial 的 32 个数据各自取平均
% mean(..., 1) 对所有 Trial (行) 取平均
mean_waveform = mean(all_trial_traces, 1, 'omitnan');

%% --- 5. 绘制时间序列波形图 ---

% 定义时间轴 (32个点)
% 刺激前12帧 (-12到-1), 刺激后20帧 (1到20)
% 刺激点在 -1 和 1 之间 (即 0 处)
time_axis_32 = [(-PRE_EVENT_FRAMES : -1), (1 : POST_EVENT_FRAMES)]; 

figure;
hold on; 

% 绘制平均波形
plot(time_axis_32, mean_waveform, 'b-', 'LineWidth', 2);

% 绘制刺激发生时刻的垂直虚线 (在时间轴上的 0 处)
xline(0, '--r', 'LineWidth', 1.5, 'DisplayName', '刺激开始点');


% 设置图表属性
title(TITLE_TEXT); % 使用动态标题
xlabel('相对刺激开始的帧数 (Time from Event, Frames)');
ylabel('神经元活动平均强度 (Delta F/F)'); % 修改Y轴标签
legend('平均波形', 'Location', 'best');
grid on;
box on;
hold off;

fprintf('\n--- 分析完成 (M%d) ---\n', mouse_id);
fprintf('有效参与平均的 Trial 数量: %d\n', valid_trials_count);
fprintf('波形图已绘制 (32个数据点，DFF归一化)。\n');