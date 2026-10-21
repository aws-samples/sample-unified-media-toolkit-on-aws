# Python 3.13 — Media Preprocess Workflow
#
################################################################################
#
# Two-stage build:
#   1. build  — compiles ffmpeg + openH264 + mxftools from source (LGPL clean)
#   2. release — Lambda image with ffmpeg and application code
#
# Default target arch: linux/arm64
# To build for amd64:
#   docker buildx build --platform linux/amd64 -t media-processing .
#
# For local testing:
#   docker run --platform linux/arm64 -it -u 0 --entrypoint /bin/bash media-processing
#
################################################################################

ARG PYTHON_VERSION=3.13.2026.02.18.08
ARG FFMPEG_VERSION=n6.1.2
ARG YASM_VERSION=1.3.0
ARG OPENH264_VERSION=2.6.0
ARG URIPARSER_VERSION=1.0.0
# Expected sha256 per downloaded artifact (CSR scan 2026-10-08) — a
# mismatch fails the build before anything is extracted or executed.
# NOTE: the GitHub archive/ digests cover codeload-generated zips,
# which GitHub has very occasionally re-generated platform-wide; a
# loud build failure + 1-line re-pin is the accepted cost.
ARG URIPARSER_SHA256=e05e445368f3cabe254a5f48796120ac4c74ee9d947e8c648505b032833d5c10
ARG YASM_SHA256=3dce6601b495f5b3d45b59f7d2492a340ee7e84b5beca17e48f862502bd5603f
ARG OPENH264_SHA256=b1ca643eadbbf16f74c86a66ab487f38691d8dbdda435683b149b2ce2c8077ec
ARG FFMPEG_SHA256=cc94e8901d9b57831a4357e0abbfe5414a987065b12e763ac036bee60993f2ac
ARG BMX_COMMIT_SHA=346ba45dfd41f24f7582460caa56c9ef47dd9b4b


#####################################################################
#
# Build Stage: compile ffmpeg + openH264 from source
#
#####################################################################
FROM public.ecr.aws/lambda/python:${PYTHON_VERSION} AS build

ARG FFMPEG_VERSION
ARG YASM_VERSION
ARG OPENH264_VERSION
ARG URIPARSER_VERSION
ARG URIPARSER_SHA256
ARG YASM_SHA256
ARG OPENH264_SHA256
ARG FFMPEG_SHA256
ARG BMX_COMMIT_SHA
ARG TARGETARCH

ARG YASM_PACKAGE=yasm-${YASM_VERSION}.tar.gz
ARG FFMPEG_PACKAGE=${FFMPEG_VERSION}.zip
ARG OPENH264_PACKAGE=v${OPENH264_VERSION}.zip
ARG URIPARSER_PACKAGE=uriparser-${URIPARSER_VERSION}.zip

ENV \
  BUILD_PREFIX=/opt \
  BUILD_BIN=/opt/bin \
  BUILD_LIB=/opt/lib

RUN echo "TARGETARCH=${TARGETARCH}, FFMPEG_VERSION=${FFMPEG_VERSION}, YASM_VERSION=${YASM_VERSION}, OPENH264_VERSION=${OPENH264_VERSION}"

# Install build dependencies
RUN \
  microdnf update -y && \
  microdnf install -y \
  autoconf \
  automake \
  bzip2 \
  bzip2-devel \
  cmake \
  freetype-devel \
  gcc \
  gcc-c++ \
  git \
  libtool \
  make \
  mercurial \
  nasm \
  openssl-devel \
  pkgconfig \
  unzip \
  zlib-devel \
  expat-devel \
  libcurl-devel \
  libuuid-devel && \
  microdnf clean all

# Build urlparser - required by BMX (mxf2raw tool)
RUN cd /var/task && \
  curl -q -L https://github.com/uriparser/uriparser/archive/refs/tags/${URIPARSER_PACKAGE} -o ${URIPARSER_PACKAGE} && \
  echo "${URIPARSER_SHA256}  ${URIPARSER_PACKAGE}" | sha256sum -c - && \
  unzip ${URIPARSER_PACKAGE} && \
  cd uriparser-*/ && \
  mkdir build && \
  cd build && \
  cmake .. \
  -DCMAKE_INSTALL_PREFIX=/opt \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_BUILD_TYPE=Release \
  -DURIPARSER_BUILD_TESTS=OFF \
  -DURIPARSER_BUILD_DOCS=OFF && \
  make -j$(nproc) && \
  make install && \
  /sbin/ldconfig

# Build BMX (mxf2raw tool)
RUN \
  cd /var/task && \
  # Pinned 2026-10-08: v1.7 (2026-05-09) vs main differs only in docs/CI
  # and a Windows-manifest CMake fix — zero Linux source changes. An
  # unpinned clone rebuilt whatever main was that day. The rev-parse
  # check below guards the pin itself: tags are mutable, the audited
  # commit is not.
  git clone --branch v1.7 --depth 1 --recurse-submodules https://github.com/ebu/bmx.git && \
  cd bmx && \
  [ "$(git rev-parse HEAD)" = "${BMX_COMMIT_SHA}" ] && \
  mkdir -p out/build && \
  cd out/build && \
  cmake ../../ \
  -DCMAKE_INSTALL_PREFIX=/opt \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_BUILD_TYPE=Release \
  -DBMX_BUILD_WITH_LIBCURL=ON \
  -DCMAKE_PREFIX_PATH=/opt && \
  cmake --build . -- -j$(nproc) && \
  cmake --install . && \
  mxf2raw --version

# Build YASM (assembler — required by ffmpeg for optimised routines)
RUN \
  cd /var/task && \
  curl -q -O -L https://www.tortall.net/projects/yasm/releases/${YASM_PACKAGE} && \
  echo "${YASM_SHA256}  ${YASM_PACKAGE}" | sha256sum -c - && \
  tar xzf ${YASM_PACKAGE} && \
  cd ./yasm-${YASM_VERSION} && \
  ./configure \
  --prefix="${BUILD_PREFIX}" \
  --bindir="${BUILD_BIN}" \
  && \
  make -j$(nproc) && \
  make install

# Build openH264
# Detect arch at build time: arm64 → aarch64, amd64 → x86_64
RUN \
  cd /var/task && \
  curl -q -O -L https://github.com/cisco/openh264/archive/refs/tags/${OPENH264_PACKAGE} && \
  echo "${OPENH264_SHA256}  ${OPENH264_PACKAGE}" | sha256sum -c - && \
  unzip ${OPENH264_PACKAGE} && \
  cd ./openh264-${OPENH264_VERSION} && \
  sed -i'.bak' -e "s|PREFIX=/usr/local|PREFIX=${BUILD_PREFIX}|g" Makefile && \
  if [ "${TARGETARCH}" = "arm64" ]; then \
  make -j$(nproc) OS=linux ARCH=aarch64; \
  else \
  make -j$(nproc) OS=linux ARCH=x86_64; \
  fi && \
  make install

# Build FFmpeg (LGPL — GPL/nonfree components explicitly disabled)
RUN \
  cd /var/task && \
  curl -q -O -L https://github.com/FFmpeg/FFmpeg/archive/refs/tags/${FFMPEG_PACKAGE} && \
  echo "${FFMPEG_SHA256}  ${FFMPEG_PACKAGE}" | sha256sum -c - && \
  unzip ${FFMPEG_PACKAGE} && \
  cd ./FFmpeg-${FFMPEG_VERSION} && \
  PKG_CONFIG_PATH="${BUILD_PREFIX}/lib/pkgconfig" ./configure \
  --prefix="${BUILD_PREFIX}" \
  --extra-cflags="-I${BUILD_PREFIX}/include -O2 -fstack-protector-strong -fpie -pie -Wl,-z,relro,-z,now -D_FORTIFY_SOURCE=2" \
  --extra-ldflags="-L${BUILD_LIB}" \
  --extra-libs=-lpthread \
  --extra-libs=-lm \
  --bindir="${BUILD_BIN}" \
  --enable-libfreetype \
  --enable-openssl \
  --enable-shared \
  --enable-pic \
  --enable-libopenh264 \
  --disable-static \
  --disable-gpl \
  --disable-nonfree \
  --disable-version3 \
  --disable-debug \
  --disable-ffplay \
  --disable-libxcb \
  --disable-libxcb-shm \
  --disable-libxcb-xfixes \
  --disable-libxcb-shape \
  --disable-lzma \
  --disable-doc \
  && \
  make -j$(nproc) && \
  make install

# Copy required system shared libraries into /opt/lib
RUN \
  cp -av /lib64/libbz2.so*           $BUILD_LIB && \
  cp -av /lib64/libfreetype.so*      $BUILD_LIB && \
  cp -av /lib64/libpng*.so*          $BUILD_LIB && \
  cp -av /lib64/libharfbuzz*.so*     $BUILD_LIB && \
  cp -av /lib64/libbrotlidec*.so*    $BUILD_LIB && \
  cp -av /lib64/libgraphite2*.so*    $BUILD_LIB && \
  cp -av /lib64/libbrotlicommon*.so* $BUILD_LIB && \
  cp -av /lib64/libz*.so*            $BUILD_LIB && \
  cp -av /lib64/libexpat.so*         $BUILD_LIB

# Rename source packages for LGPL distribution compliance
RUN \
  cd /var/task && \
  mv -v ${FFMPEG_PACKAGE}    source_ffmpeg.zip && \
  mv -v ${OPENH264_PACKAGE}  source_openh264.zip

# Create /opt zip (ffmpeg binaries + libs) for reference / layer use
RUN \
  cd /opt && \
  zip -r --symlinks /var/task/package.zip .


#####################################################################
#
# Release Stage: Lambda runtime image
#
#####################################################################
FROM public.ecr.aws/lambda/python:${PYTHON_VERSION}

# ── Copy ffmpeg build artifacts from build stage ──────────────────
COPY --from=build /opt/bin/        /opt/bin/
COPY --from=build /opt/lib/        /opt/lib/

# ── Copy LGPL source packages (distribution compliance) ───────────
COPY --from=build /var/task/source_ffmpeg.zip   /opt/sources/source_ffmpeg.zip
COPY --from=build /var/task/source_openh264.zip /opt/sources/source_openh264.zip

# ── Make ffmpeg/ffprobe available on PATH ─────────────────────────
ENV PATH="/opt/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/lib:${LD_LIBRARY_PATH}"

# ── System deps ───────────────────────────────────────────────────
RUN \
  microdnf install -y libjpeg-turbo && \
  microdnf clean all

# # ── Python deps ───────────────────────────────────────────────────
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────
COPY requirements.txt lambda_function.py ./
COPY lib/               ./lib/
COPY states/            ./states/

RUN pip install --no-cache-dir -r requirements.txt

USER 1001

# ── Lambda handler ────────────────────────────────────────────────
CMD [ "lambda_function.lambda_handler" ]
