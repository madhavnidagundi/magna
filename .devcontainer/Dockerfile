# Use the official Ubuntu 24.04 as the base image
FROM ubuntu:24.04

LABEL org.opencontainers.image.source=https://github.com/MAGNA-Global/embassy-rs

# Install necessary packages, retrying with fresh metadata if a mirror or proxy
# serves a stale package (for example, an apt "Hash Sum mismatch").
RUN set -eux; \
    for attempt in 1 2 3; do \
        apt-get clean; \
        rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/partial/*; \
        if apt-get -o Acquire::Retries=5 \
                   -o Acquire::http::No-Cache=true \
                   -o Acquire::https::No-Cache=true \
                   -o Acquire::http::Pipeline-Depth=0 update && \
           apt-get -o Acquire::Retries=5 \
                   -o Acquire::http::No-Cache=true \
                   -o Acquire::https::No-Cache=true \
                   -o Acquire::http::Pipeline-Depth=0 \
                   install -y --no-install-recommends --fix-missing \
                   wget ninja-build openocd xz-utils ca-certificates sudo git gh minicom \
                   libncurses6 libncursesw6 curl can-utils iproute2 cmake build-essential \
                   lsb-release gpg protobuf-compiler nasm libssl-dev pkg-config \
                   openssh-client libclang-dev llvm-dev clang libudev-dev; then \
            break; \
        fi; \
        if [ "${attempt}" -eq 3 ]; then exit 1; fi; \
    done; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /opt/

# Set build-time arguments for user configuration
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

# Remove the default 'ubuntu' user (UID 1000) that ships with Ubuntu 24.04,
# then create our own user at the desired UID/GID.
RUN mkdir -p /etc/sudoers.d && \
    (userdel -r ubuntu 2>/dev/null || true) && \
    (groupdel ubuntu 2>/dev/null || true) && \
    groupadd --force --gid ${USER_GID} ${USERNAME} && \
    useradd -l --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash && \
    echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# Switch to the newly created user
USER ${USERNAME}

# Set environment variables
ENV LANG=C.UTF-8

# Install Rust with clippy and rustfmt components, and security tools
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . $HOME/.cargo/env && \
    rustup component add clippy rustfmt && \
    curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash && \
    cargo binstall -y cargo-audit cargo-deny

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]
