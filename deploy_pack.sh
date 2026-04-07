#!/bin/bash
# deploy_pack.sh — 自動打包自上次部署以來的變更檔案
#
# 用法:
#   ./deploy_pack.sh          # 打包上次部署後的所有變更
#   ./deploy_pack.sh --hierarchical  # 保留目錄結構，可在 Linux 直接解壓覆蓋
#
# 流程:
#   1. 比對 git tag "deployed" 與目前 HEAD 的差異
#   2. 將變更檔案打包成 zip (檔名含日期)
#   3. 更新 "deployed" tag 到目前 commit
#
# 首次使用: git tag deployed HEAD  (標記目前已部署的版本)

set -e

TAG="deployed"
DATE=$(date +%Y%m%d_%H%M%S)
OUTDIR="deploy_packages"
HIERARCHICAL=false

if [[ "$1" == "--hierarchical" ]]; then
    HIERARCHICAL=true
fi

# 檢查 tag 是否存在
if ! git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "❌ Tag '$TAG' 不存在。首次使用請先執行:"
    echo "   git tag deployed HEAD"
    exit 1
fi

# 取得變更檔案列表 (排除已刪除的檔案)
CHANGED=$(git diff --name-only --diff-filter=d "$TAG"..HEAD)

if [ -z "$CHANGED" ]; then
    echo "✅ 沒有新的變更需要部署 ($TAG 已是最新)"
    exit 0
fi

echo "📦 自 $(git log -1 --format='%h %s' $TAG) 以來的變更:"
echo "$CHANGED" | while read f; do echo "  - $f"; done
echo ""

mkdir -p "$OUTDIR"
ZIPNAME="deploy_${DATE}.zip"

if [ "$HIERARCHICAL" = true ]; then
    # 保留目錄結構 — 在 Linux 目標目錄直接 unzip -o 即可覆蓋
    echo "$CHANGED" | xargs zip -r "$OUTDIR/$ZIPNAME"
else
    # 平鋪 — 所有檔案放在 zip 根目錄 (舊行為)
    echo "$CHANGED" | xargs zip -j "$OUTDIR/$ZIPNAME"
fi

echo ""
echo "✅ 已打包 → $OUTDIR/$ZIPNAME"
echo ""

# 顯示被刪除的檔案 (需要在 Linux 手動刪除)
DELETED=$(git diff --name-only --diff-filter=D "$TAG"..HEAD)
if [ -n "$DELETED" ]; then
    echo "⚠️  以下檔案已刪除，請在 Linux 手動移除:"
    echo "$DELETED" | while read f; do echo "  - $f"; done
    echo ""
fi

# 詢問是否更新 tag
read -p "🏷️  更新 deployed tag 到目前 HEAD? (y/N) " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    git tag -f "$TAG" HEAD
    echo "✅ Tag 'deployed' 已更新到 $(git log -1 --format='%h')"
else
    echo "⏭️  Tag 未更新，下次打包會包含相同變更"
fi
