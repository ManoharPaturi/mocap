# Run this in PowerShell AS ADMINISTRATOR to allow Python through firewall
# Right-click PowerShell -> Run as Administrator

# Allow inbound connections on port 5000 (Discovery)
New-NetFirewallRule -DisplayName "MoCap Multi-Camera Port 5000" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow

# Allow inbound connections on port 5001 (Data)
New-NetFirewallRule -DisplayName "MoCap Multi-Camera Port 5001" -Direction Inbound -LocalPort 5001 -Protocol TCP -Action Allow

Write-Host "✅ Firewall rules added for ports 5000 and 5001"
