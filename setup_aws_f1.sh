#!/bin/bash
# AWS F1 Quick Setup Script for ZETAGRID-FPGA
# Run this on your LOCAL machine (Windows PowerShell or WSL)

set -e

echo "🔷 ZETAGRID-FPGA: AWS F1 Setup Wizard"
echo "======================================"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Installing..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
fi

# Configure AWS
echo ""
echo "📋 Step 1: Configure AWS Credentials"
echo "You'll need:"
echo "  - AWS Access Key ID"
echo "  - AWS Secret Access Key"
echo "  - Default region (use: us-east-1)"
echo ""
aws configure

# Create SSH Key
echo ""
echo "🔑 Step 2: Creating SSH Key..."
if [ ! -f ~/.ssh/zetagrid-fpga-key ]; then
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/zetagrid-fpga-key -N ""
    echo "✅ SSH Key created: ~/.ssh/zetagrid-fpga-key"
else
    echo "⚠️  SSH Key already exists"
fi

# Import Key to AWS
echo ""
echo "📤 Step 3: Uploading SSH Key to AWS..."
aws ec2 import-key-pair \
    --key-name zetagrid-fpga-key \
    --public-key-material fileb://~/.ssh/zetagrid-fpga-key.pub \
    --region us-east-1 2>/dev/null || echo "⚠️  Key already imported"

# Request F1 Quota Increase
echo ""
echo "📊 Step 4: Checking F1 Instance Quota..."
QUOTA=$(aws service-quotas get-service-quota \
    --service-code ec2 \
    --quota-code L-74FC7D96 \
    --region us-east-1 \
    --query 'Quota.Value' --output text 2>/dev/null || echo "0")

if [ "$QUOTA" == "0" ]; then
    echo "⚠️  F1 quota is 0. You need to request increase:"
    echo ""
    echo "   1. Go to: https://console.aws.amazon.com/servicequotas"
    echo "   2. Search: 'Running On-Demand F instances'"
    echo "   3. Request: 8 vCPUs (= 1 F1.2xlarge instance)"
    echo "   4. Wait 24-48 hours for approval"
    echo ""
    echo "   Or run this command:"
    echo "   aws service-quotas request-service-quota-increase \\"
    echo "       --service-code ec2 \\"
    echo "       --quota-code L-74FC7D96 \\"
    echo "       --desired-value 8 \\"
    echo "       --region us-east-1"
    echo ""
    read -p "Press Enter after requesting quota increase..."
else
    echo "✅ F1 Quota: $QUOTA vCPUs (sufficient)"
fi

# Launch Instance
echo ""
echo "🚀 Step 5: Launching F1 Instance..."
echo "This will cost ~$1.65/hour (covered by free credits)"
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id ami-0c55b159cbfafe1f0 \
        --instance-type f1.2xlarge \
        --key-name zetagrid-fpga-key \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ZETAGRID-FPGA}]' \
        --region us-east-1 \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    echo "✅ Instance launched: $INSTANCE_ID"
    echo "⏳ Waiting for instance to start..."
    
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region us-east-1
    
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --region us-east-1 \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    echo ""
    echo "======================================"
    echo "✅ F1 Instance Ready!"
    echo "======================================"
    echo "Instance ID: $INSTANCE_ID"
    echo "Public IP:   $PUBLIC_IP"
    echo ""
    echo "Connect with:"
    echo "  ssh -i ~/.ssh/zetagrid-fpga-key centos@$PUBLIC_IP"
    echo ""
    echo "💡 Next: Run setup_fpga.sh on the instance"
    echo "======================================"
    
    # Save connection info
    echo "export FPGA_IP=$PUBLIC_IP" > fpga_connection.sh
    echo "export FPGA_INSTANCE=$INSTANCE_ID" >> fpga_connection.sh
    chmod +x fpga_connection.sh
fi

echo ""
echo "🎉 Setup Complete!"
