#!/bin/bash

# MP4 音频提取转换为 WAV 脚本
# 使用方法: ./mp4_to_wav.sh [选项] 输入文件.mp4 [输出文件.wav]

# 默认参数
SAMPLE_RATE="44100"
BIT_DEPTH="16"
CHANNELS="2"
OVERWRITE=false

# 显示帮助信息
function show_help {
    echo "用法: $0 [选项] 输入文件.mp4 [输出文件.wav]"
    echo "选项:"
    echo "  -h, --help            显示此帮助信息"
    echo "  -r, --rate <Hz>       设置采样率 (默认: 44100)"
    echo "  -b, --bit-depth <位>  设置位深度 (默认: 16)"
    echo "  -c, --channels <数量> 设置声道数 (默认: 2)"
    echo "  -f, --force           强制覆盖已存在的输出文件"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -r|--rate)
            SAMPLE_RATE="$2"
            shift 2
            ;;
        -b|--bit-depth)
            BIT_DEPTH="$2"
            shift 2
            ;;
        -c|--channels)
            CHANNELS="$2"
            shift 2
            ;;
        -f|--force)
            OVERWRITE=true
            shift
            ;;
        *)
            # 非选项参数，视为输入/输出文件
            if [[ -z "$INPUT_FILE" ]]; then
                INPUT_FILE="$1"
            elif [[ -z "$OUTPUT_FILE" ]]; then
                OUTPUT_FILE="$1"
            else
                echo "错误: 太多参数" >&2
                show_help
            fi
            shift
            ;;
    esac
done

# 检查输入文件是否指定
if [[ -z "$INPUT_FILE" ]]; then
    echo "错误: 必须指定输入文件" >&2
    show_help
fi

# 检查输入文件是否存在
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "错误: 输入文件 '$INPUT_FILE' 不存在" >&2
    exit 1
fi

# 检查输入文件是否为 MP4 格式（简单检查扩展名）
if [[ "${INPUT_FILE,,}" != *.mp4 ]]; then
    echo "警告: 输入文件 '$INPUT_FILE' 似乎不是 MP4 文件" >&2
    read -p "是否继续? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 如果未指定输出文件，自动生成
if [[ -z "$OUTPUT_FILE" ]]; then
    OUTPUT_FILE="${INPUT_FILE%.*}.wav"
fi

# 检查输出文件是否已存在
if [[ -f "$OUTPUT_FILE" && "$OVERWRITE" = false ]]; then
    echo "错误: 输出文件 '$OUTPUT_FILE' 已存在。使用 -f 选项强制覆盖。" >&2
    exit 1
fi

# 检查 ffmpeg 是否已安装
if ! command -v ffmpeg &> /dev/null; then
    echo "错误: ffmpeg 未安装。请先安装 ffmpeg。" >&2
    exit 1
fi

# 执行转换
echo "开始转换: '$INPUT_FILE' -> '$OUTPUT_FILE'"
echo "参数: 采样率=${SAMPLE_RATE}Hz, 位深度=${BIT_DEPTH}位, 声道=${CHANNELS}"

ffmpeg_cmd=(
    ffmpeg -i "$INPUT_FILE" 
    -vn  # 不包含视频
    -acodec pcm_s${BIT_DEPTH}le  # PCM 有符号整数，little-endian
    -ar "$SAMPLE_RATE"  # 采样率
    -ac "$CHANNELS"  # 声道数
    -hide_banner -loglevel warning  # 减少输出信息
)

# 添加覆盖选项（如果需要）
if [[ "$OVERWRITE" = true ]]; then
    ffmpeg_cmd+=(-y)
fi

ffmpeg_cmd+=("$OUTPUT_FILE")

# 执行命令
"${ffmpeg_cmd[@]}"

# 检查转换是否成功
if [[ $? -eq 0 ]]; then
    echo "转换成功!"
    # 获取原始音频和转换后音频的信息
    original_duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT_FILE")
    converted_duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUTPUT_FILE")
    
    echo "原始音频时长: $(printf "%.2f" "$original_duration") 秒"
    echo "转换后音频时长: $(printf "%.2f" "$converted_duration") 秒"
    
    # 检查时长是否基本一致（误差在0.5秒内）
    duration_diff=$(echo "$original_duration - $converted_duration" | bc)
    duration_diff=${duration_diff#-}  # 取绝对值
    if (( $(echo "$duration_diff > 0.5" | bc -l) )); then
        echo "警告: 转换前后音频时长差异较大，可能存在问题。" >&2
    fi
else
    echo "转换失败!" >&2
    exit 1
fi    