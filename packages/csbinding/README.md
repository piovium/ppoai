# Genius Invokation TCG (C# binding)

`packages/csbinding` is the .NET wrapper around the native C binding.

It exposes managed wrappers for the common simulator objects:

- `GiTcg`
- `CreateParam`
- `State`
- `Game`
- `IPlayer`

The wrapper delegates all simulator work to the C binding and uses generated protobuf classes for player IO.

## Build From This Repository

The C# package depends on two prerequisites:

1. the C binding must already be built and installed under `packages/cbinding/install`
2. `NativeMethods.cs` must be generated from the C header

Typical local sequence:

```sh
cd packages/cbinding
cmake -B build -G Ninja
cmake --build build --config Release
cmake --install build --config Release --prefix install

cd ../csbinding-gen
cargo run -- ../cbinding/include/gitcg/gitcg.h -o ../csbinding/GiTcg/NativeMethods.cs

cd ../csbinding
dotnet build -c Release
```

`GiTcg.csproj` copies the native library from `../../cbinding/install/lib` or `../../cbinding/install/bin` into the build output.

## Runtime Model

The C# API is a managed wrapper over native handles:

- `CreateParam` builds initial deck/state parameters
- `State` wraps state creation and JSON round-tripping
- `Game` wraps player registration, stepping, and state access
- `IPlayer` receives protobuf notifications and request objects

The wrapper bootstraps the native library and thread-local environment through `ObjBase`.

## IO Model

The callback flow mirrors the C API:

- native code delivers protobuf request bytes
- the wrapper parses them into generated protobuf classes
- `IPlayer` returns the typed response
- the wrapper serializes the response back to bytes for the native layer

## Sample

See [`GiTcgTest/Program.cs`](./GiTcgTest/Program.cs) for the current end-to-end example.

## Maturity Note

This is currently a secondary parity path. Use Python first for AI, self-play, and training-oriented work; use C# when a .NET integration is the actual target.
