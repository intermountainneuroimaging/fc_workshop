#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 01_install_aws_cli.sh
# Install AWS CLI v2 into scratch.
#
# Why this exists:
#   The OpenNeuro Midnight Scan Club data are downloaded from public S3.
#   You do not need an AWS account, but you do need the AWS CLI command.
###############################################################################

export USERNAME="${USERNAME:-jade6100}"
export SOFTWARE_DIR="${SOFTWARE_DIR:-/scratch/alpine/${USERNAME}/software}"
export BIN_DIR="${BIN_DIR:-${SOFTWARE_DIR}/bin}"
export AWS_INSTALL_DIR="${AWS_INSTALL_DIR:-${SOFTWARE_DIR}/aws-cli}"

mkdir -p "${SOFTWARE_DIR}" "${BIN_DIR}"
cd "${SOFTWARE_DIR}"

echo "Installing AWS CLI into:"
echo "  ${AWS_INSTALL_DIR}"
echo "Binary link will be:"
echo "  ${BIN_DIR}/aws"
echo

# Remove old installer files so unzip does not ask:
# replace aws/README.md? [y]es, [n]o, [A]ll...
rm -rf aws awscliv2.zip

echo "Downloading AWS CLI v2 installer..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip

echo "Unzipping AWS CLI installer..."
unzip -q awscliv2.zip

echo "Running AWS installer..."
./aws/install \
  --install-dir "${AWS_INSTALL_DIR}" \
  --bin-dir "${BIN_DIR}" \
  --update

export PATH="${BIN_DIR}:${PATH}"

echo
echo "AWS CLI installed:"
which aws
aws --version
