# rust-webgl2-example

Rust + WebAssembly + WebGL 2.0 Demo

Demo: https://likr.github.io/rust-webgl2-example

# How to build

```console
$ source path/to/emsdk/emsdk_env.sh
$ export CLANG_PATH=`which clang`
$ cargo build --release --target=wasm32-unknown-emscripten
```
