#!/usr/bin/env bash
# claude.sh — run Claude Code inside a bubblewrap sandbox.
#
# ============================================================================
# QUICK START
# ============================================================================
#
# Prerequisites:
#   - bubblewrap installed:   sudo apt install bubblewrap
#   - Claude Code installed under $HOME/.local/share/claude/versions/ (the
#     native installer's default layout: each version is either a single
#     executable file named after the version, e.g. "2.1.143", or a directory
#     containing a "claude" binary).
#
# Install (pick one):
#   1. Symlink it onto your PATH:
#        ln -s "$PWD/claude.sh" "$HOME/.local/bin/claude"
#      Ensure "$HOME/.local/bin" comes before any other claude in PATH.
#
#   2. Source it as a shell function:
#        echo 'source /path/to/claude.sh' >> ~/.bashrc
#      Then re-open your shell and use `claude` as normal.
#
# Use:
#   cd /path/to/some/project
#   conda activate myenv     # optional; the active env is forwarded
#   claude                   # launches sandboxed claude with CWD writable
#   claude update            # self-update; runs the native updater on the
#                            # host, unsandboxed (see UPDATING CLAUDE CODE)
#   claude --ssh github.com  # + git push / ssh to github.com through a
#                            # per-session ssh-agent; the private key never
#                            # enters the sandbox (see SSH ACCESS)
#
# ============================================================================
# SANDBOX LAYOUT (what claude sees)
# ============================================================================
#
#   /usr, /etc, /opt, /lib*, /bin, /sbin    read-only (system)
#   /usr/local                              read-only; if it's a symlink off
#                                           /usr (e.g. to /mnt/.../local) the
#                                           real target is auto-exposed too
#   /tmp, /var/tmp, /run, /proc, /dev       fresh, ephemeral
#   $HOME                                   read-only tmpfs — only these are
#                                           visible inside it:
#     $HOME/.claude                         read-write (claude's state dir)
#     $HOME/.claude.json                    read-write (claude's config file)
#     $CONDA_PREFIX  (if set)               read-only by default; the base
#                                           conda install (where conda/mamba
#                                           live) is also bound so those
#                                           commands work. Set
#                                           CLAUDE_SANDBOX_CONDA_WRITE=1 to
#                                           bind both read-write — required
#                                           for `mamba install`/`pip install`
#                                           into the active env.
#     $CWD                                  read-write (unless CWD == $HOME,
#                                           in which case nothing extra is
#                                           bound — refuse to expose all of
#                                           $HOME)
#   network                                 By default: --share-net plus
#                                           HTTPS_PROXY pointing at a host-
#                                           side mitmproxy with an allowlist
#                                           (see "Network modes" below).
#                                           Override per-invocation via
#                                           CLAUDE_SANDBOX_NET=open|strict|none.
#
# Environment is cleared by default. The default allowlist forwards locale
# (LANG/LC_*/TZ), color hints, Anthropic config (ANTHROPIC_API_KEY etc.),
# standard proxy vars, TLS CA overrides, and CUDA vars (CUDA_HOME,
# CUDAToolkit_ROOT, LD_LIBRARY_PATH). Anything else (AWS_*, GH_TOKEN,
# KUBECONFIG, SSH_AUTH_SOCK, ...) stays outside.
#
# ============================================================================
# OPTIONAL KNOBS (set in the calling shell)
# ============================================================================
#
#   CLAUDE_SANDBOX_RO=/path/a:/path/b   extra read-only paths (colon-sep)
#   CLAUDE_SANDBOX_RW=/path/c           extra read-write paths (colon-sep)
#   CLAUDE_SANDBOX_PASSENV="A B C"      extra env vars to forward (space-
#                                       or colon-separated names)
#   CLAUDE_SANDBOX_CONDA_WRITE=1        bind the active conda env (and its
#                                       base install) read-write, so claude
#                                       can `mamba install`, `pip install`,
#                                       etc. Default 0 (read-only) is safer.
#   CLAUDE_SANDBOX_NET=proxy|strict|open|none  network mode (default: proxy)
#       proxy  : --share-net plus HTTPS_PROXY pointing at a host-side
#                mitmproxy that enforces ~/.config/claude-sandbox/allowlist.txt.
#                Any tool that honors HTTPS_PROXY (claude, curl, git, pip,
#                npm, conda, ...) is filtered. Tools that bypass HTTPS_PROXY
#                with raw sockets, or talk to localhost/LAN directly, are
#                NOT filtered. This is the working default on Ubuntu 24.04.
#       strict : --unshare-net + slirp4netns + HTTPS_PROXY. Closes the raw-
#                socket loophole and blocks localhost/LAN. Does not currently
#                work on Ubuntu 24.04 — slirp4netns can't enter bwrap's
#                unprivileged netns without root/setuid helpers. Kept for
#                future use once that's solved (rootlesskit, pasta, ...).
#       open   : --share-net, no proxy. Debug only.
#       none   : no network at all (offline work).
#
# Sensitive locations (~/.ssh, ~/.gnupg, ~/.aws, ~/.config/{gcloud,sops},
# ~/.kube, ~/.docker, ~/.password-store) are refused both as CWD and as
# CLAUDE_SANDBOX_RW targets. Paths that resolve to "/" or "$HOME" are
# refused outright.
#
# ============================================================================
# UPDATING CLAUDE CODE
# ============================================================================
#
# `claude update` (alias `upgrade`) and `claude install [target]` are
# intercepted by this wrapper and run with the newest installed binary
# directly on the host, NOT inside the sandbox. Inside the sandbox they
# cannot work: the proxy allowlist blocks downloads.claude.ai, the versions
# directory is not bound, and DISABLE_AUTOUPDATER=1 is set (so a sandboxed
# session never tries to self-update on its own either).
#
# The native updater installs the new release under
# $HOME/.local/share/claude/versions/<version>. This wrapper always runs the
# highest version found there, so the update takes effect on the next
# launch. As of Claude Code 2.1.263 the updater refuses to overwrite a
# launcher at $HOME/.local/bin/claude that is not its own versions/ symlink
# (it logs "Not replacing ..." and still reports success), so the wrapper
# stays on PATH. Should an installer re-point the launcher anyway, the
# wrapper restores it right after the subcommand finishes.
#
# Because the launcher is not installer-managed, Claude Code skips its
# automatic cleanup of old versions (~215 MB each). Prune by hand:
#   ls ~/.local/share/claude/versions/     # then rm the ones you don't need
#
# ============================================================================
# SSH ACCESS (git push, remote hosts)
# ============================================================================
#
# By default a session has NO SSH credentials: nothing under ~/.ssh is bound
# and no agent socket is forwarded, so it cannot push, cannot ssh anywhere,
# and cannot post to a remote. That is the "locked" tier and needs no flag.
#
# Flags (claude.sh's own; they must come BEFORE claude's arguments and are
# stripped before claude sees the command line):
#
#   --ssh HOST            allow SSH to HOST. Repeatable. HOST is anything ssh
#                         accepts: a hostname, user@host, or an alias from
#                         ~/.ssh/config (HostName/User/Port/IdentityFile are
#                         honoured -- resolution is done on the host with
#                         `ssh -G`).
#   --ssh-unrestricted    allow SSH anywhere the key is accepted. Mutually
#                         exclusive with --ssh HOST. The escape hatch.
#   --ssh-key PATH        private key to use. Default: the first readable
#                         IdentityFile ssh itself would try for that host
#                         (config-aware, FIDO *_sk keys skipped).
#   --ssh-timeout LIFE    expire the loaded key after LIFE (seconds, or
#                         e.g. 30m, 2h). Off unless given.
#
# How it works: claude.sh starts a FRESH ssh-agent for this session only,
# loads the key into it, binds just that agent's unix socket into the
# sandbox and exports SSH_AUTH_SOCK. ssh/git inside ask the host agent to
# sign; the private key itself is never visible in the sandbox. When the
# session ends the agent is killed and its socket removed. An agent orphaned
# by an uncatchable kill (SIGKILL of the wrapper) is reaped by the next --ssh
# launch's janitor, which skips still-running sessions, so it is safe with
# concurrent sessions. Session state lives under
# ${XDG_RUNTIME_DIR:-/tmp}/claude-sandbox-ssh.<uid>/.
#
# Per-host enforcement: with --ssh HOST the key is loaded with an OpenSSH
# destination constraint (ssh-add -h, OpenSSH >= 8.9). The AGENT then
# refuses to sign for any other host -- verified: for a non-permitted host
# the key is not even offered. This matters because SSH is a raw socket and
# shares the host network namespace in "proxy" mode, so the mitmproxy
# allowlist cannot see or filter it; the constraint is the control.
#
# User pinning: the constraint carries a user only when you asked for one,
# i.e. you wrote user@host, or ~/.ssh/config sets User for that alias. So
# `--ssh github.com` permits any user (git@ included); `--ssh git@github.com`
# permits only git@. Non-22 ports are handled ([host]:port, as known_hosts
# stores them).
#
# Requirements and limits:
#   - The host's key must already be in ~/.ssh/known_hosts on the host
#     (agent constraints are keyed by host key). If it is not, claude.sh
#     refuses with a hint; connect once from a host shell (`ssh HOST`).
#     This is deliberate: no trust-on-first-use is done for you.
#   - ~/.ssh/known_hosts and ~/.ssh/config are bound READ-ONLY into the
#     sandbox (public data). Nothing else under ~/.ssh is. `Include` files
#     referenced from config are not bound; keep aliases in the main file.
#   - An encrypted key prompts for its passphrase once, on the host tty, at
#     launch. No askpass/DISPLAY is needed. A non-tty launch cannot prompt.
#   - Constraints are host-level, not operation-level: within a permitted
#     host the SSH tunnel is opaque, so e.g. a force-push cannot be blocked.
#   - --ssh-unrestricted loads the key with no constraint: a compromised
#     session can then authenticate as you to any host that trusts the key
#     for as long as the session lives (mitigate with --ssh-timeout).
#   - Git identity is separate: see the repo-local [user] hint at launch.
#
# ============================================================================
# ENVIRONMENT-SPECIFIC SETUP NOTES
# ============================================================================
#
# Ubuntu 24.04+: unprivileged user namespaces are blocked by AppArmor.
#   You'll see:
#     bwrap: setting up uid map: Permission denied
#     bwrap: No permissions to create new namespace, ...
#   The recommended fix is to add an AppArmor profile that grants bwrap the
#   `userns` rule (keeps the global restriction on for everything else):
#
#     sudo tee /etc/apparmor.d/bwrap >/dev/null <<'EOF'
#     abi <abi/4.0>,
#     include <tunables/global>
#
#     profile bwrap /usr/bin/bwrap flags=(unconfined) {
#       userns,
#       include if exists <local/bwrap>
#     }
#     EOF
#     sudo apparmor_parser -r /etc/apparmor.d/bwrap
#
#   Verify with:
#     bwrap --ro-bind / / --unshare-user --unshare-pid -- /bin/true && echo OK
#
#   Quicker but coarser alternative — disable the restriction system-wide:
#     sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0
#     echo 'kernel.apparmor_restrict_unprivileged_userns = 0' \
#       | sudo tee /etc/sysctl.d/60-userns.conf
#
# CUDA under /usr/local: this works automatically. If /usr/local is a
#   symlink (e.g. to /mnt/<disk>/local), the symlink target is auto-bound
#   read-only so paths like /usr/local/cuda resolve inside the sandbox.
#   CUDA_HOME, CUDAToolkit_ROOT, and LD_LIBRARY_PATH are forwarded by
#   default.
#
# Conda/mamba: activate your environment BEFORE running claude. The script
#   reads $CONDA_PREFIX from your shell, binds it read-only inside the
#   sandbox, and forwards PATH and CONDA_* variables so the env stays
#   activated. There is no in-sandbox `conda activate` step.
#
# Running from $HOME: deliberately constrained. Nothing under $HOME is
#   bound except the explicit allowlist (.claude, .claude.json, the conda
#   env if it lives in $HOME, anything you list in CLAUDE_SANDBOX_R{O,W}).
#   $HOME itself is a read-only tmpfs, so `touch $HOME/foo` fails. Use a
#   project subdirectory instead, or set CLAUDE_SANDBOX_RW=... for the
#   specific subtrees you want exposed.
#
# ============================================================================
# NETWORK MODES — ONE-TIME SETUP
# ============================================================================
#
# The default network mode (proxy) needs three things on the host:
#
#   1. slirp4netns + mitmproxy installed:
#        sudo apt install slirp4netns mitmproxy
#
#   2. mitmproxy's CA cert trusted system-wide (so TLS interception works
#      for claude, node, curl, git, pip, ...):
#        mitmdump --listen-port 0 &   # one-time, to materialise the CA
#        sleep 1; kill %1
#        sudo cp ~/.mitmproxy/mitmproxy-ca-cert.pem \
#                /usr/local/share/ca-certificates/mitmproxy.crt
#        sudo update-ca-certificates
#
#   3. The user systemd service that runs mitmdump with the allowlist
#      addon enabled and started:
#        systemctl --user daemon-reload
#        systemctl --user enable --now claude-mitmproxy.service
#      (Unit file: ~/.config/systemd/user/claude-mitmproxy.service; addon
#      and allowlist live in ~/.config/claude-sandbox/.)
#
# Then every claude run goes through mitmproxy, which enforces the
# allowlist at ~/.config/claude-sandbox/allowlist.txt. Blocked hosts are
# logged to ~/.config/claude-sandbox/blocked.log — tail it to see what
# claude is reaching for, and add hosts as needed (no restart required).
#
# Verify:
#   systemctl --user status claude-mitmproxy
#   tail -f ~/.config/claude-sandbox/blocked.log
#
# Inside a sandboxed session:
#   ! curl -sI https://api.anthropic.com    # allowed
#   ! curl -sI https://example.com          # 403 from mitmproxy
#   ! curl -sI http://127.0.0.1:5432        # connection refused (slirp4netns)
#
# ============================================================================
# TROUBLESHOOTING
# ============================================================================
#
#   "no versions under .../versions"
#       Claude Code isn't installed at the expected path, or the versions
#       directory is empty. Check $HOME/.local/share/claude/versions/.
#
#   "bwrap: setting up uid map: Permission denied"
#   "bwrap: No permissions to create new namespace ..."
#       See the Ubuntu 24.04 AppArmor section above.
#
#   "Can't bind mount ... on /newroot/usr/local: ..."
#       /usr/local is a symlink whose target isn't being exposed. The script
#       handles this automatically now; if you still hit it, expose the
#       target manually:  CLAUDE_SANDBOX_RO=/path/to/real/local claude
#
#   "Claude configuration file not found at: /home/USER/.claude.json"
#       The script binds this file automatically; make sure you re-symlinked
#       /didn't drop the latest claude.sh.
#
#   "API Error: Unable to connect to API (FailedToOpenSocket)" or DNS errors
#       Usually caused by /etc/resolv.conf being a symlink into /run on
#       systemd-resolved systems. The script binds the resolved target
#       automatically; verify with: readlink -f /etc/resolv.conf
#
#   "claude.sh: slirp4netns not installed"
#       Either install it (sudo apt install slirp4netns) or set
#       CLAUDE_SANDBOX_NET=open to bypass (loses network sandboxing).
#
#   Every HTTPS request fails with cert errors (UNABLE_TO_VERIFY_LEAF_SIGNATURE,
#   SSL_ERROR_BAD_CERT_DOMAIN, ...)
#       mitmproxy's CA cert isn't trusted. Run:
#         sudo cp ~/.mitmproxy/mitmproxy-ca-cert.pem \
#                 /usr/local/share/ca-certificates/mitmproxy.crt
#         sudo update-ca-certificates
#       If only Node/claude breaks (other tools work), the system CA bundle
#       isn't being read by Node — point it at the bundle explicitly:
#         CLAUDE_SANDBOX_PASSENV="NODE_EXTRA_CA_CERTS" \
#         NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt claude
#
#   "claude.sh: bwrap did not report a child PID"
#       bwrap failed to set up the sandbox before slirp4netns could attach.
#       Run bwrap manually with the same args to see its real error.
#
#   "setns(CLONE_NEWNET): Operation not permitted" from slirp4netns
#       On Ubuntu 24.04+, the kernel.apparmor_restrict_unprivileged_userns
#       hardening also blocks slirp4netns. Grant it userns the same way we
#       did for bwrap:
#         sudo tee /etc/apparmor.d/slirp4netns >/dev/null <<'EOF'
#         abi <abi/4.0>,
#         include <tunables/global>
#         profile slirp4netns /usr/bin/slirp4netns flags=(unconfined) {
#           userns,
#           include if exists <local/slirp4netns>
#         }
#         EOF
#         sudo apparmor_parser -r /etc/apparmor.d/slirp4netns
#       If you still see the error after that, slirp4netns may be too old
#       (need 1.0+ for --userns-path), or the bwrap child PID has exited
#       before slirp could attach.
#
#   Allowed hosts still return 403 from mitmproxy
#       Check the addon is loaded:  systemctl --user status claude-mitmproxy
#       Check the allowlist format: bare hostnames; leading "." for subdomains.
#       Watch the journal:  journalctl --user -u claude-mitmproxy -f
#
#   Tool inside claude can't find an API token (AWS, GH, ...)
#       By design: only an explicit allowlist is forwarded. Re-export per
#       session with:  CLAUDE_SANDBOX_PASSENV="GH_TOKEN" claude
#
#   Tool inside claude can't reach a file outside CWD that you need
#       Expose it explicitly:  CLAUDE_SANDBOX_RO=/some/dir claude
#       (Or CLAUDE_SANDBOX_RW for writable.)
#
#   "refusing to bind sensitive directory '...' as CWD"
#       You ran claude from a known-secret directory (~/.ssh etc.). cd
#       somewhere else.
#
#   "claude.sh: no trusted host key for '<host>' ..." (with --ssh)
#       The host is not in ~/.ssh/known_hosts on the host side. Connect once
#       from a HOST shell (`ssh <host>`) to record it, then relaunch.
#
#   "Host key verification failed" inside the sandbox
#       Same cause: known_hosts is bound read-only, so unknown hosts cannot
#       be added from inside. Trust the host from a host shell first.
#
#   "Permission denied (publickey)" to host B while running with --ssh A
#       Working as designed: the agent only signs for hosts named in --ssh.
#       Add `--ssh B` (or use --ssh-unrestricted).
#
# ============================================================================
# SECURITY CAVEATS
# ============================================================================
#
#   - Network: in the default "proxy" mode, every HTTPS_PROXY-aware tool
#     (claude, curl, git, pip, npm, conda) is filtered against
#     ~/.config/claude-sandbox/allowlist.txt by a host-side mitmproxy.
#     Tools that bypass HTTPS_PROXY (raw sockets, direct localhost calls)
#     are NOT filtered — they can still reach your dev DBs, your LAN, etc.
#     "strict" mode would close that gap via slirp4netns but is currently
#     non-functional on Ubuntu 24.04 (see Network modes section above).
#   - $HOME/.claude is read-write and holds Anthropic OAuth/API tokens.
#     A compromised claude inside the sandbox can exfiltrate them; the
#     sandbox cannot prevent this.
#   - $CONDA_PREFIX (if set) is read-only but readable, including any
#     secrets stashed in etc/conda/activate.d/*.sh.
#   - $CWD (and .git, .env, etc. inside it) is read-write to claude.
#   - bwrap relies on unprivileged user namespaces. There is no seccomp
#     filter or resource cap configured here.
#   - `claude update|upgrade|install` run the native binary on the host,
#     unsandboxed, with your full environment (see UPDATING CLAUDE CODE).
#   - --ssh HOST / --ssh-unrestricted forward a per-session ssh-agent socket.
#     The key stays on the host, but while the session lives the sandbox can
#     authenticate to the permitted host(s) as you (any host, if
#     unrestricted). SSH egress is not filtered by the proxy allowlist.

claude() (
  set -euo pipefail

  # ---- claude.sh's own flags (must precede claude's arguments) ----
  # See "SSH ACCESS" in the header. Parsing stops at the first token that is
  # not one of ours; everything from there on is passed to claude untouched.
  local -a _ssh_hosts=()
  local _ssh_unrestricted=0 _ssh_key="" _ssh_timeout="" _ssh_want=0
  while (( $# > 0 )); do
    case "$1" in
      --ssh)
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] \
          || { echo "claude.sh: --ssh needs a host (hostname, user@host, or ~/.ssh/config alias)" >&2; return 2; }
        _ssh_hosts+=( "$2" ); shift 2 ;;
      --ssh=?*)           _ssh_hosts+=( "${1#--ssh=}" ); shift ;;
      --ssh-unrestricted) _ssh_unrestricted=1; shift ;;
      --ssh-key)
        [[ $# -ge 2 && -n "$2" ]] || { echo "claude.sh: --ssh-key needs a path" >&2; return 2; }
        _ssh_key="$2"; shift 2 ;;
      --ssh-key=?*)       _ssh_key="${1#--ssh-key=}"; shift ;;
      --ssh-timeout)
        [[ $# -ge 2 && -n "$2" ]] || { echo "claude.sh: --ssh-timeout needs a lifetime (seconds, or e.g. 30m)" >&2; return 2; }
        _ssh_timeout="$2"; shift 2 ;;
      --ssh-timeout=?*)   _ssh_timeout="${1#--ssh-timeout=}"; shift ;;
      *) break ;;
    esac
  done
  (( ${#_ssh_hosts[@]} > 0 || _ssh_unrestricted )) && _ssh_want=1
  if (( _ssh_unrestricted && ${#_ssh_hosts[@]} > 0 )); then
    echo "claude.sh: --ssh-unrestricted and --ssh HOST are mutually exclusive" >&2; return 2
  fi
  if (( ! _ssh_want )) && [[ -n "$_ssh_key$_ssh_timeout" ]]; then
    echo "claude.sh: --ssh-key / --ssh-timeout need --ssh HOST or --ssh-unrestricted" >&2; return 2
  fi
  if [[ -n "$_ssh_timeout" && ! "$_ssh_timeout" =~ ^[0-9]+[smhdwSMHDW]?$ ]]; then
    echo "claude.sh: --ssh-timeout: bad lifetime '$_ssh_timeout' (seconds, or e.g. 30m, 2h)" >&2; return 2
  fi
  if [[ -n "$_ssh_key" && ! -r "$_ssh_key" ]]; then
    echo "claude.sh: --ssh-key: cannot read '$_ssh_key'" >&2; return 2
  fi

  versions_dir="${HOME}/.local/share/claude/versions"
  [[ -d "$versions_dir" ]] || { echo "claude.sh: missing $versions_dir" >&2; return 1; }

  # Each "version" under $versions_dir is either a single executable file
  # named after the version (e.g. 2.1.143) or a directory containing a
  # `claude` binary. Pick the highest version-sorted entry of either kind.
  latest=$(
    find "$versions_dir" -mindepth 1 -maxdepth 1 \( -type f -o -type d -o -type l \) \
         -printf '%f\n' 2>/dev/null \
      | sort -V | tail -n1
  )
  [[ -n "$latest" ]] || { echo "claude.sh: no versions under $versions_dir" >&2; return 1; }

  claude_bin=""
  if [[ -x "$versions_dir/$latest" && ! -d "$versions_dir/$latest" ]]; then
    claude_bin="$versions_dir/$latest"
  else
    for cand in "$versions_dir/$latest/claude" "$versions_dir/$latest/bin/claude"; do
      [[ -x "$cand" ]] && { claude_bin="$cand"; break; }
    done
  fi
  [[ -n "$claude_bin" ]] || { echo "claude.sh: no executable found for version '$latest'" >&2; return 1; }

  # ---- host-side subcommands: update / upgrade / install ----
  # See "UPDATING CLAUDE CODE" in the header. These run the newest installed
  # binary directly on the host with the caller's environment; the sandbox
  # would block the download and has no writable versions directory. After
  # the subcommand, restore $HOME/.local/bin/claude if the installer moved
  # it off this script.
  case "${1:-}" in
    update|upgrade|install)
      (( _ssh_want )) && echo "claude.sh: --ssh* flags are ignored for '$1'" >&2
      local launcher="$HOME/.local/bin/claude" self before after rc=0
      self="$(readlink -f -- "${BASH_SOURCE[0]}")"
      before="$(readlink -f -- "$launcher" 2>/dev/null || true)"
      echo "claude.sh: running '$latest $*' on the host (unsandboxed)" >&2
      "$claude_bin" "$@" || rc=$?
      after="$(readlink -f -- "$launcher" 2>/dev/null || true)"
      if [[ "$before" == "$self" && "$after" != "$self" ]]; then
        ln -sfn -- "$self" "$launcher"
        echo "claude.sh: installer re-pointed $launcher -> $after; restored -> $self" >&2
      fi
      echo "claude.sh: installed versions: $(find "$versions_dir" -mindepth 1 -maxdepth 1 \
              \( -type f -o -type d -o -type l \) -printf '%f\n' 2>/dev/null | sort -V | tr '\n' ' ')" >&2
      return "$rc"
      ;;
  esac

  cwd="$(pwd -P)"
  mkdir -p "$HOME/.claude"
  # Claude also keeps top-level config in ~/.claude.json. Ensure it exists
  # so we can bind it (bwrap refuses missing sources).
  [[ -e "$HOME/.claude.json" ]] || : > "$HOME/.claude.json"

  # Refuse to launch from directories that almost certainly hold secrets.
  case "$cwd" in
    "$HOME"/.ssh|"$HOME"/.ssh/*|\
    "$HOME"/.gnupg|"$HOME"/.gnupg/*|\
    "$HOME"/.aws|"$HOME"/.aws/*|\
    "$HOME"/.config/gcloud|"$HOME"/.config/gcloud/*|\
    "$HOME"/.config/sops|"$HOME"/.config/sops/*|\
    "$HOME"/.kube|"$HOME"/.kube/*|\
    "$HOME"/.docker|"$HOME"/.docker/*|\
    "$HOME"/.password-store|"$HOME"/.password-store/*)
      echo "claude.sh: refusing to bind sensitive directory '$cwd' as CWD." >&2
      return 1 ;;
  esac

  # ----- git identity hint -----
  # ~/.gitconfig is not bound into the sandbox, so a *global* git identity is
  # invisible inside it and commits fail with "Author identity unknown". The
  # supported fix is a repo-local identity: .git/config travels with the repo
  # (bound read-write as CWD), so a [user] section there is enough on its own
  # -- git reads local before global and needs nothing else. If this repo has
  # none, say so. Setting it must happen from a HOST shell to copy the values
  # out of ~/.gitconfig, which is not visible in here either.
  if command -v git >/dev/null \
     && git -C "$cwd" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if [[ -z "$(git -C "$cwd" config --local user.name  2>/dev/null)" \
       || -z "$(git -C "$cwd" config --local user.email 2>/dev/null)" ]]; then
      local _gitroot
      _gitroot="$(git -C "$cwd" rev-parse --show-toplevel 2>/dev/null || echo "$cwd")"
      echo "claude.sh: no repo-local git identity; commits inside the sandbox will fail" >&2
      echo "           (~/.gitconfig is not bound in). Set it once, from a HOST shell:" >&2
      echo "             git -C \"$_gitroot\" config user.name  \"\$(git config --global user.name)\"" >&2
      echo "             git -C \"$_gitroot\" config user.email \"\$(git config --global user.email)\"" >&2
    fi
  fi

  # When CWD is $HOME we deliberately do *not* bind it: that would expose
  # the entire home directory. Only the explicit bindings below apply.
  bind_cwd=1
  if [[ "$cwd" == "$HOME" ]]; then
    bind_cwd=0
    echo "claude.sh: CWD is \$HOME — not binding it. Only \$HOME/.claude," \
         "the active conda env, and CLAUDE_SANDBOX_R{O,W} paths will be visible." >&2
  fi

  # Likewise, refuse user-supplied RW paths that pull the rug out from
  # under the sandbox or expose well-known secret stores.
  if [[ -n "${CLAUDE_SANDBOX_RW:-}" ]]; then
    local IFS=':' _rw=() _p
    read -ra _rw <<< "$CLAUDE_SANDBOX_RW"
    for _p in "${_rw[@]}"; do
      [[ -z "$_p" ]] && continue
      _real="$(readlink -f -- "$_p" 2>/dev/null || echo "$_p")"
      case "$_real" in
        /|"$HOME")
          echo "claude.sh: refusing CLAUDE_SANDBOX_RW='$_p' (would expose all of '$_real' read-write)." >&2
          return 1 ;;
        "$HOME"/.ssh*|"$HOME"/.gnupg*|"$HOME"/.aws*|\
        "$HOME"/.config/gcloud*|"$HOME"/.config/sops*|\
        "$HOME"/.kube*|"$HOME"/.docker*|"$HOME"/.password-store*)
          echo "claude.sh: refusing CLAUDE_SANDBOX_RW='$_p' (sensitive path '$_real')." >&2
          return 1 ;;
      esac
    done
  fi

  args=(
    # Minimal system view (all read-only). Note: anything under /usr (incl.
    # /usr/local when it's a real directory) is already covered by the /usr
    # bind. When /usr/local is a symlink elsewhere we expose its real target
    # below so the symlink resolves inside the sandbox (needed for CUDA).
    --ro-bind     /usr        /usr
    --ro-bind-try /etc        /etc
    --ro-bind-try /opt        /opt
    --ro-bind-try /lib        /lib
    --ro-bind-try /lib64      /lib64
    --ro-bind-try /bin        /bin
    --ro-bind-try /sbin       /sbin

    # Ephemeral pseudo-filesystems.
    --proc   /proc
    --dev    /dev
    --tmpfs  /tmp
    --tmpfs  /var/tmp
    --tmpfs  /run

    # Blank $HOME — we punch through only what's needed below.
    --tmpfs "$HOME"

    # Isolation. Network is left unshared here; we re-introduce it below
    # according to CLAUDE_SANDBOX_NET (default: through a host-side mitmproxy
    # allowlist via slirp4netns).
    --unshare-all
    --die-with-parent
    --new-session

    # Drop the parent environment; we re-export an explicit allowlist below.
    --clearenv

    # Always-on bindings.
    --bind     "$HOME/.claude"      "$HOME/.claude"
    --bind     "$HOME/.claude.json" "$HOME/.claude.json"
    --ro-bind  "$claude_bin"        "$claude_bin"
  )

  # If /usr/local is a symlink off /usr (e.g. to a separate filesystem),
  # expose its real target read-only so the symlink resolves inside the
  # sandbox. Needed for things like CUDA installed under /usr/local.
  if [[ -L /usr/local ]]; then
    _usrlocal_real="$(readlink -f /usr/local 2>/dev/null || true)"
    if [[ -n "$_usrlocal_real" && -d "$_usrlocal_real" ]]; then
      args+=( --ro-bind "$_usrlocal_real" "$_usrlocal_real" )
    fi
  fi

  # On systemd-resolved systems (Ubuntu default), /etc/resolv.conf is a
  # symlink into /run/systemd/resolve/, and we tmpfs'd /run. Bind the real
  # target through the tmpfs so DNS works inside the sandbox.
  if [[ -L /etc/resolv.conf ]]; then
    _resolv_real="$(readlink -f /etc/resolv.conf 2>/dev/null || true)"
    if [[ -n "$_resolv_real" && -e "$_resolv_real" && "$_resolv_real" != /etc/* ]]; then
      args+=( --ro-bind "$_resolv_real" "$_resolv_real" )
    fi
  fi

  # ---- environment allowlist ----
  # Always-set basics.
  args+=(
    --setenv HOME "$HOME"
    --setenv USER "${USER:-$(id -un)}"
    --setenv PATH "$PATH"
    --setenv TERM "${TERM:-xterm-256color}"
    # Never let the sandboxed claude try to self-update. Downloads are
    # blocked by the proxy allowlist and the versions directory is not
    # bound, so the attempt could only fail; `claude update` (run on the
    # host by this wrapper) is the supported path. Note: the legacy
    # `autoUpdates: false` in ~/.claude.json is ignored for native installs.
    --setenv DISABLE_AUTOUPDATER 1
  )
  # Variables that are safe-to-forward IF set in the caller's environment.
  # Locale + display niceties, Anthropic config, and standard proxy vars.
  # Note: secret tokens for *other* services (AWS, GitHub, K8s, npm, ...)
  # are NOT in this list and stay outside the sandbox.
  _claude_default_passenv=(
    LANG LC_ALL LC_CTYPE LC_MESSAGES LC_NUMERIC LC_TIME LC_COLLATE
    TZ COLORTERM NO_COLOR FORCE_COLOR
    ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL
    ANTHROPIC_MODEL ANTHROPIC_SMALL_FAST_MODEL
    HTTP_PROXY HTTPS_PROXY NO_PROXY ALL_PROXY
    http_proxy https_proxy no_proxy all_proxy
    SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
    CUDA_HOME CUDAToolkit_ROOT LD_LIBRARY_PATH
  )
  # Extra names the user explicitly opts in via CLAUDE_SANDBOX_PASSENV
  # (space- or colon-separated).
  _claude_extra_passenv=()
  if [[ -n "${CLAUDE_SANDBOX_PASSENV:-}" ]]; then
    # Split on whitespace or ':'.
    local _raw="${CLAUDE_SANDBOX_PASSENV//:/ }"
    # shellcheck disable=SC2206
    _claude_extra_passenv=( $_raw )
  fi
  for _var in "${_claude_default_passenv[@]}" "${_claude_extra_passenv[@]}"; do
    [[ -z "$_var" ]] && continue
    # Indirect lookup; skip if unset (vs. set-empty).
    if [[ -n "${!_var+x}" ]]; then
      args+=( --setenv "$_var" "${!_var}" )
    fi
  done
  if (( bind_cwd )); then
    args+=( --bind "$cwd" "$cwd" )
  fi
  args+=( --chdir "$cwd" )

  # Active conda/mamba environment. Mode is controlled by
  # CLAUDE_SANDBOX_CONDA_WRITE:
  #   unset/0  : env is read-only — claude can run tools from it but cannot
  #              install/uninstall packages (`mamba install` will fail).
  #   1        : env (and the base conda install) is read-write — claude
  #              can mutate the environment, including installing software.
  # The base conda install (where the conda/mamba binaries live) is bound
  # in the same mode so commands like `mamba install` and `conda list`
  # actually find their executables and package cache.
  if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX" ]]; then
    if [[ "${CLAUDE_SANDBOX_CONDA_WRITE:-0}" == "1" ]]; then
      _conda_mode="--bind"
    else
      _conda_mode="--ro-bind"
    fi

    # Find the base conda install (where condabin/, bin/conda, bin/mamba,
    # and the package cache pkgs/ live). Try several signals, in order.
    _conda_base=""
    if [[ -n "${MAMBA_ROOT_PREFIX:-}" && -d "${MAMBA_ROOT_PREFIX}" ]]; then
      _conda_base="$MAMBA_ROOT_PREFIX"
    elif [[ -n "${CONDA_ROOT_PREFIX:-}" && -d "${CONDA_ROOT_PREFIX}" ]]; then
      _conda_base="$CONDA_ROOT_PREFIX"
    elif [[ -n "${CONDA_PYTHON_EXE:-}" && -x "${CONDA_PYTHON_EXE}" ]]; then
      _conda_base="$(dirname "$(dirname "$CONDA_PYTHON_EXE")")"
    elif [[ "$CONDA_PREFIX" == */envs/* ]]; then
      _conda_base="${CONDA_PREFIX%/envs/*}"
    fi

    args+=(
      "$_conda_mode" "$CONDA_PREFIX" "$CONDA_PREFIX"
      --setenv CONDA_PREFIX "$CONDA_PREFIX"
    )
    [[ -n "${CONDA_DEFAULT_ENV:-}" ]] && args+=( --setenv CONDA_DEFAULT_ENV "$CONDA_DEFAULT_ENV" )
    [[ -n "${CONDA_SHLVL:-}"       ]] && args+=( --setenv CONDA_SHLVL       "$CONDA_SHLVL"       )
    [[ -n "${MAMBA_ROOT_PREFIX:-}" ]] && args+=( --setenv MAMBA_ROOT_PREFIX "$MAMBA_ROOT_PREFIX" )
    [[ -n "${CONDA_ROOT_PREFIX:-}" ]] && args+=( --setenv CONDA_ROOT_PREFIX "$CONDA_ROOT_PREFIX" )
    [[ -n "${CONDA_PYTHON_EXE:-}"  ]] && args+=( --setenv CONDA_PYTHON_EXE  "$CONDA_PYTHON_EXE"  )

    # Bind the base prefix separately, but only if it's distinct from
    # $CONDA_PREFIX (otherwise we'd try to bind the same path twice and
    # bwrap would error).
    if [[ -n "$_conda_base" && "$_conda_base" != "$CONDA_PREFIX" \
          && -d "$_conda_base" ]]; then
      args+=( "$_conda_mode" "$_conda_base" "$_conda_base" )
    fi
  fi

  # User-supplied extra paths.
  _claude_add_paths() {
    local mode="$1" raw="${2:-}"
    [[ -z "$raw" ]] && return 0
    local IFS=':' parts=() p
    read -ra parts <<< "$raw"
    for p in "${parts[@]}"; do
      [[ -z "$p" ]] && continue
      if [[ ! -e "$p" ]]; then
        echo "claude.sh: warning — skipping missing path '$p'" >&2
        continue
      fi
      args+=( "$mode" "$p" "$p" )
    done
  }
  _claude_add_paths --ro-bind "${CLAUDE_SANDBOX_RO:-}"
  _claude_add_paths --bind    "${CLAUDE_SANDBOX_RW:-}"

  # Lock $HOME itself down: the tmpfs is read-only, so claude cannot create
  # files directly in $HOME. The sub-binds above (.claude, .claude.json, the
  # conda env, $CWD, RW knobs) keep their own rw/ro semantics, so this only
  # blocks writes to $HOME proper — not to anything we intentionally exposed.
  # Must come AFTER every bind whose target is under $HOME, since bwrap may
  # need to create intermediate directories on the tmpfs.
  # ----- SSH access: per-session agent with destination constraints -----
  # See "SSH ACCESS" in the header. Runs on the HOST, before the sandbox
  # exists: resolves each --ssh spec through ~/.ssh/config, checks the host
  # key is already trusted, starts a fresh agent, loads the key(s) with
  # destination constraints, and binds only the agent socket (plus the
  # public known_hosts/config files) into the sandbox.
  _ssh_agent_pid="" _ssh_dir=""
  _claude_ssh_cleanup() {
    # Kill only if the recorded PID is still an ssh-agent -- guards against a
    # PID that was already reaped and recycled to an unrelated process.
    if [[ -n "${_ssh_agent_pid:-}" \
       && "$(cat "/proc/${_ssh_agent_pid}/comm" 2>/dev/null)" == ssh-agent ]]; then
      kill "$_ssh_agent_pid" 2>/dev/null
    fi
    [[ -n "${_ssh_dir:-}" ]] && rm -rf -- "$_ssh_dir"
    _ssh_agent_pid="" _ssh_dir=""
    return 0
  }
  _claude_ssh_pick_key() {
    # $1 = `ssh -G` output. Print the first readable IdentityFile ssh itself
    # would try (config-aware). FIDO *_sk keys need a provider; skip them.
    local cand
    while read -r cand; do
      cand="${cand/#\~/$HOME}"
      [[ "$cand" == *_sk ]] && continue
      [[ -r "$cand" ]] && { printf '%s\n' "$cand"; return 0; }
    done < <(awk '$1=="identityfile"{print $2}' <<<"$1")
    return 1
  }
  _claude_ssh_sweep() {
    # Janitor for orphans left by an uncatchable kill (SIGKILL of the wrapper
    # runs no trap). A session dir is orphaned iff its owner process is gone:
    # we compare the recorded PID *and* its start-time, so a recycled PID (same
    # number, newer start-time) is never mistaken for the live owner. Live
    # sessions are always skipped, so this is safe to run with any number of
    # concurrent claude.sh sessions on the host.
    local base="$1" d opid ostart now agent
    local _ng=0; shopt -q nullglob && _ng=1; shopt -s nullglob
    for d in "$base"/session.*; do
      [[ -d "$d" ]] || continue
      [[ -r "$d/owner.id" ]] || continue          # unstamped => mid-setup; leave it
      read -r opid ostart < "$d/owner.id" 2>/dev/null || continue
      if [[ -n "$opid" && -r "/proc/$opid/stat" ]]; then
        now="$(awk '{print $22}' "/proc/$opid/stat" 2>/dev/null || true)"
        [[ -n "$now" && "$now" == "$ostart" ]] && continue   # owner alive -> skip
      fi
      # Orphan: drop keys via the socket first (addresses the agent by socket,
      # immune to PID reuse), then guard-kill the recorded agent, then remove.
      [[ -S "$d/agent.sock" ]] && SSH_AUTH_SOCK="$d/agent.sock" ssh-add -D >/dev/null 2>&1 || true
      if [[ -r "$d/agent.pid" ]] && read -r agent < "$d/agent.pid" 2>/dev/null \
         && [[ -n "$agent" && "$(cat "/proc/$agent/comm" 2>/dev/null)" == ssh-agent ]]; then
        kill "$agent" 2>/dev/null || true
      fi
      rm -rf -- "$d" 2>/dev/null || true
    done
    (( _ng )) || shopt -u nullglob
    return 0
  }
  _claude_ssh_setup() {
    local kh="$HOME/.ssh/known_hosts" cfg="$HOME/.ssh/config"
    command -v ssh-agent >/dev/null && command -v ssh-add >/dev/null && command -v ssh-keygen >/dev/null \
      || { echo "claude.sh: ssh-agent/ssh-add/ssh-keygen not found (install openssh-client)" >&2; return 1; }
    [[ -r "$kh" ]] || { echo "claude.sh: $kh not readable; --ssh needs pre-trusted host keys there" >&2; return 1; }

    declare -A _by_key=()          # key file -> space-separated constraints
    local spec g host user port khname key localuser
    localuser="$(id -un)"
    for spec in "${_ssh_hosts[@]}"; do
      [[ "$spec" != -* ]] || { echo "claude.sh: bad --ssh host '$spec'" >&2; return 1; }
      g="$(ssh -G "$spec" 2>/dev/null)" || { echo "claude.sh: cannot resolve '$spec' (ssh -G failed)" >&2; return 1; }
      host="$(awk '$1=="hostname"{print $2; exit}' <<<"$g")"
      user="$(awk '$1=="user"    {print $2; exit}' <<<"$g")"
      port="$(awk '$1=="port"    {print $2; exit}' <<<"$g")"
      # Pin the user only when asked for: user@ in the spec, or set by config.
      [[ "$spec" != *@* && "$user" == "$localuser" ]] && user=""
      khname="$host"; [[ "$port" != 22 ]] && khname="[$host]:$port"
      if ! ssh-keygen -F "$khname" -f "$kh" >/dev/null 2>&1; then
        echo "claude.sh: no trusted host key for '$khname' (--ssh $spec) in $kh." >&2
        echo "           Connect once from a HOST shell to record it:   ssh $spec" >&2
        return 1
      fi
      key="$_ssh_key"
      [[ -n "$key" ]] || key="$(_claude_ssh_pick_key "$g")" \
        || { echo "claude.sh: no readable private key for '$spec'; pass --ssh-key PATH" >&2; return 1; }
      _by_key[$key]+=" ${user:+$user@}$khname"
    done
    if (( _ssh_unrestricted )); then
      key="$_ssh_key"
      [[ -n "$key" ]] || key="$(_claude_ssh_pick_key "$(ssh -G claude-sandbox.invalid 2>/dev/null)")" \
        || { echo "claude.sh: no readable private key found under ~/.ssh; pass --ssh-key PATH" >&2; return 1; }
      _by_key[$key]=""
    fi

    # Per-user base dir, pinned (not $TMPDIR) so the janitor finds siblings
    # deterministically. Sweep prior orphans before adding this session.
    local base="${XDG_RUNTIME_DIR:-/tmp}/claude-sandbox-ssh.$(id -u)"
    mkdir -p "$base" 2>/dev/null && chmod 700 "$base" \
      || { echo "claude.sh: cannot create session base $base" >&2; return 1; }
    _claude_ssh_sweep "$base" || true

    # A fresh agent for this session only. Its socket is the one secret-
    # bearing thing bound in; the key files themselves never are.
    _ssh_dir="$(mktemp -d "$base/session.XXXXXX")" || return 1
    chmod 700 "$_ssh_dir"
    # Liveness stamp for the janitor: this subshell's PID (it runs bwrap and
    # holds the trap) plus its start-time, written BEFORE the agent exists so a
    # concurrent sweep never sees a dir of ours unstamped.
    printf '%s %s\n' "$BASHPID" \
      "$(awk '{print $22}' "/proc/$BASHPID/stat" 2>/dev/null || echo 0)" > "$_ssh_dir/owner.id"
    local sock="$_ssh_dir/agent.sock"
    eval "$(ssh-agent -s -a "$sock")" >/dev/null \
      || { echo "claude.sh: ssh-agent failed to start" >&2; return 1; }
    _ssh_agent_pid="$SSH_AGENT_PID"
    printf '%s\n' "$_ssh_agent_pid" > "$_ssh_dir/agent.pid"
    trap '_claude_ssh_cleanup' EXIT INT TERM HUP

    local c; local -a add
    for key in "${!_by_key[@]}"; do
      add=()
      [[ -n "$_ssh_timeout" ]] && add+=( -t "$_ssh_timeout" )
      if [[ -n "${_by_key[$key]}" ]]; then
        for c in ${_by_key[$key]}; do add+=( -h "$c" ); done   # no spaces in constraints
        add+=( -H "$kh" )
        echo "claude.sh: session agent: $key constrained to:${_by_key[$key]}${_ssh_timeout:+  (lifetime $_ssh_timeout)}" >&2
      else
        echo "claude.sh: session agent: $key UNRESTRICTED${_ssh_timeout:+  (lifetime $_ssh_timeout)}" >&2
      fi
      SSH_AUTH_SOCK="$sock" ssh-add "${add[@]}" "$key" \
        || { echo "claude.sh: ssh-add failed for $key" >&2; return 1; }
    done

    args+=(
      --bind    "$sock" "$sock"
      --setenv  SSH_AUTH_SOCK "$sock"
      --ro-bind "$kh" "$kh"
    )
    [[ -r "$cfg" ]] && args+=( --ro-bind "$cfg" "$cfg" )
    return 0
  }
  if (( _ssh_want )); then
    _claude_ssh_setup || { _claude_ssh_cleanup; return 1; }
  fi

  # With a session agent bash must outlive bwrap to kill the agent and remove
  # its socket, so it cannot exec. Without one, exec as before.
  _claude_launch() {
    if [[ -n "${_ssh_agent_pid:-}" ]]; then
      local rc=0
      bwrap "$@" || rc=$?
      _claude_ssh_cleanup
      return "$rc"
    fi
    exec bwrap "$@"
  }

  args+=( --remount-ro "$HOME" )

  # ----- Network mode -----
  # proxy  : (default) --share-net plus HTTPS_PROXY pointing at the host-side
  #          mitmproxy (claude-mitmproxy.service on 127.0.0.1:8888). All
  #          HTTPS_PROXY-aware tools (claude, curl, git, pip, npm, conda, ...)
  #          go through the allowlist at
  #          ~/.config/claude-sandbox/allowlist.txt. Limitation: tools that
  #          bypass HTTPS_PROXY by opening raw sockets (or talking to
  #          localhost directly) are NOT filtered.
  # strict : --unshare-net + slirp4netns + HTTPS_PROXY. Closes the raw-socket
  #          loophole and also blocks localhost/LAN. Currently does not work
  #          on Ubuntu 24.04 (slirp4netns cannot setns into bwrap's unshared
  #          netns from outside without root/setuid helpers); kept here for
  #          when that's solvable (e.g. rootlesskit, pasta, or a fixed
  #          slirp4netns codepath).
  # open   : --share-net with no proxy. Use only for debugging.
  # none   : net fully unshared. Useful for offline work.
  case "${CLAUDE_SANDBOX_NET:-proxy}" in
    open)
      args+=( --share-net )
      _claude_launch "${args[@]}" -- "$claude_bin" "$@"
      ;;
    none)
      _claude_launch "${args[@]}" -- "$claude_bin" "$@"
      ;;
    proxy)
      args+=(
        --share-net
        --setenv HTTP_PROXY  "http://127.0.0.1:8888"
        --setenv HTTPS_PROXY "http://127.0.0.1:8888"
        --setenv http_proxy  "http://127.0.0.1:8888"
        --setenv https_proxy "http://127.0.0.1:8888"
        --setenv NO_PROXY    ""
        --setenv no_proxy    ""
      )
      _claude_launch "${args[@]}" -- "$claude_bin" "$@"
      ;;
    strict)
      command -v slirp4netns >/dev/null || {
        echo "claude.sh: slirp4netns not installed (apt install slirp4netns)." >&2
        return 1
      }
      args+=(
        --setenv HTTP_PROXY  "http://10.0.2.2:8888"
        --setenv HTTPS_PROXY "http://10.0.2.2:8888"
        --setenv http_proxy  "http://10.0.2.2:8888"
        --setenv https_proxy "http://10.0.2.2:8888"
        --setenv NO_PROXY    ""
        --setenv no_proxy    ""
      )

      # Coordinate bwrap and slirp4netns: bwrap writes its child PID to fd 19
      # then blocks on fd 18 until slirp4netns has configured the namespace
      # and signals ready via --ready-fd 18. Strict mode also requires
      # mitmdump to be listening on a host address slirp's gateway routes
      # to (NOT 127.0.0.1 — see ../systemd/user/claude-mitmproxy.service).
      local _tmp _bwrap_pid _slirp_pid _child_pid _rc
      _tmp=$(mktemp -d -t claude-sandbox.XXXXXX)
      trap '
        [[ -n "${_slirp_pid:-}" ]] && kill "$_slirp_pid" 2>/dev/null
        [[ -n "${_bwrap_pid:-}" ]] && kill "$_bwrap_pid" 2>/dev/null
        _claude_ssh_cleanup
        rm -rf "$_tmp"
      ' EXIT INT TERM
      mkfifo "$_tmp/info" "$_tmp/ready"
      exec 18<>"$_tmp/ready"

      bwrap --info-fd 19 --block-fd 18 "${args[@]}" -- "$claude_bin" "$@" \
            19>"$_tmp/info" &
      _bwrap_pid=$!

      _child_pid=$(
        timeout 10 python3 -c \
          'import json,sys; print(json.load(sys.stdin)["child-pid"])' \
          <"$_tmp/info" 2>/dev/null
      ) || {
        echo "claude.sh: bwrap did not report a child PID; aborting." >&2
        wait "$_bwrap_pid" 2>/dev/null
        exit $?
      }

      slirp4netns --configure --mtu=65520 --disable-host-loopback \
                  --userns-path "/proc/$_child_pid/ns/user" \
                  --ready-fd 18 "$_child_pid" tap0 >/dev/null &
      _slirp_pid=$!

      wait "$_bwrap_pid"; _rc=$?
      kill "$_slirp_pid" 2>/dev/null
      wait "$_slirp_pid" 2>/dev/null || true
      exit "$_rc"
      ;;
    *)
      echo "claude.sh: unknown CLAUDE_SANDBOX_NET='$CLAUDE_SANDBOX_NET'" \
           "(expected proxy|strict|open|none)" >&2
      return 1
      ;;
  esac
)

# When this file is executed (e.g. via a symlink in $PATH), invoke the
# function with the caller's args. When it is sourced, only define it.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  claude "$@"
fi
