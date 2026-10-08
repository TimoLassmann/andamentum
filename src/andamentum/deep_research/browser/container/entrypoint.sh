#!/bin/sh
# Runs as root: install the egress firewall, then drop to the browser user.
# Allowed out: loopback, replies, DNS to the resolvers, the public internet.
# Rejected: every private, loopback, link-local, shared and multicast range,
# which covers the host (host.containers.internal), the VM gateways and the LAN.
set -e
iptables -P OUTPUT ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
for ns in $(awk '/^nameserver/ {print $2}' /etc/resolv.conf | grep -E '^[0-9.]+$'); do
  iptables -A OUTPUT -d "$ns" -p udp --dport 53 -j ACCEPT
  iptables -A OUTPUT -d "$ns" -p tcp --dport 53 -j ACCEPT
done
for net in 0.0.0.0/8 10.0.0.0/8 100.64.0.0/10 127.0.0.0/8 169.254.0.0/16 172.16.0.0/12 192.0.0.0/24 192.168.0.0/16 198.18.0.0/15 224.0.0.0/3; do
  iptables -A OUTPUT -d "$net" -j REJECT
done
ip6tables -P OUTPUT DROP 2>/dev/null || true
ip6tables -A OUTPUT -o lo -j ACCEPT 2>/dev/null || true
exec setpriv --reuid 10001 --regid 10001 --clear-groups --bounding-set -all --inh-caps -all --no-new-privs env HOME=/home/browser /browser.sh
