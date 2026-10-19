# Rust implmentation

This is a rust implementation of the main part of the simulation.  The python
grate prep command must be run first on the main yaml to do cross-section and
layer interpolation, and set up area/P/ng/Bwet interpolators.  The output of
this is saved in a directory.  Then the rust implementation can take over

## Prep

```
grate prep input.yaml afolder
```

## Run rust command on afolder

```
cargo run --release afolder
```


