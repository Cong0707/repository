sudo curl -s https://install.zerotier.com/ | sudo bash
sudo zerotier-cli join 6ab565387a0d0396
sudo curl -sfL https://rancher-mirror.rancher.cn/k3s/k3s-install.sh | INSTALL_K3S_MIRROR=cn K3S_URL=https://172.30.25.108:6443 K3S_TOKEN="K100ffe807e81c499b22a61ca54d7abe473a0d00e0e5dc7bc932bf59de175ad1ed4::server:1869897287675e5eadeaba7ab627e7f2" sh -
