#!/bin/bash
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}$*${NC}"; }
err()  { echo -e "${RED}$*${NC}"; }
warn() { echo -e "${YELLOW}$*${NC}"; }

echo "================================"
echo "Python虚拟环境管理"
echo "================================"

export PATH="$HOME/.local/bin:$PATH"
# 检查 uv 是否已安装
if ! command -v uv &> /dev/null; then
    err "uv 未安装"
    echo "正在安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if ! command -v uv &> /dev/null; then
        err "uv 安装失败，请手动安装: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    ok "uv 安装成功"
fi

# 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo ""
    echo "虚拟环境不存在，正在创建..."
    uv venv
    ok "虚拟环境创建成功"
else
    ok "虚拟环境已存在"
fi

source .venv/bin/activate

# 检测 CUDA 版本并安装对应 torch
echo ""
echo "正在检测 CUDA 版本..."
TORCH_INDEX_URL=""
CUDA_VERSION=""

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
elif command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
fi

if [ -n "$CUDA_VERSION" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    CU_VERSION="cu${CUDA_MAJOR}${CUDA_MINOR}"
    ok "检测到 CUDA ${CUDA_VERSION}"
    case "$CU_VERSION" in
        cu118|cu121|cu122|cu124)
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
            ok "将使用 torch+cu126 版本 (兼容 CUDA ${CUDA_VERSION} 驱动)"
            ;;
        cu126|cu128)
            TORCH_INDEX_URL="https://download.pytorch.org/whl/${CU_VERSION}"
            ok "将使用 torch+${CU_VERSION} 版本"
            ;;
        *)
            warn "警告: CUDA ${CUDA_VERSION} 不在已知支持列表，使用默认 torch"
            ;;
    esac
else
    err "未检测到 CUDA，请确认已安装 NVIDIA 驱动及 CUDA"
    exit 1
fi

echo ""
echo "正在安装依赖..."

if [ -n "$TORCH_INDEX_URL" ]; then
    TORCH_CU=$(basename "$TORCH_INDEX_URL")
    echo "正在安装 torch (${TORCH_CU})..."
    uv pip install "torch>=2.7.0" --index-url "$TORCH_INDEX_URL"
    ok "torch 安装完成"
fi

if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
    ok "依赖(requirements.txt)安装完成"
else
    warn "未找到 requirements.txt"
fi

if [ -f "pyproject.toml" ]; then
    uv pip install -e .
    ok "依赖(pyproject.toml)安装完成"
else
    warn "未找到 pyproject.toml"
fi

echo ""
echo "================================"
ok "虚拟环境已激活！"
echo "================================"
echo ""
echo "当前环境信息："
echo "  Python: $(python --version)"
echo "  位置: $VIRTUAL_ENV"
echo ""
echo "已安装的包："
uv pip list
echo ""
echo "================================"
echo "下一步："
echo "================================"
echo ""
echo "1. 激活虚拟环境："
echo "    source .venv/bin/activate"
