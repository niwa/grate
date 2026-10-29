import sys
import numpy as np
import xarray as xr


def compare_netcdf(file1, file2, rtol=1e-6, atol=1e-8):
    differences = 0

    def report(message):
        nonlocal differences
        differences += 1
        print(message)

    with xr.open_dataset(file1) as a, xr.open_dataset(file2) as b:
        # Global attributes
        compare_attrs("global attributes", a.attrs, b.attrs, report)

        # Dimensions
        for name in sorted(set(a.sizes) | set(b.sizes)):
            if name not in a.sizes:
                report(f"Dimension {name!r} missing from file 1")
            elif name not in b.sizes:
                report(f"Dimension {name!r} missing from file 2")
            elif a.sizes[name] != b.sizes[name]:
                report(f"Dimension {name!r}: {a.sizes[name]} vs {b.sizes[name]}")

        # Variables, including coordinates
        for name in sorted(set(a.variables) | set(b.variables)):
            if name not in a.variables:
                report(f"Variable {name!r} missing from file 1")
                continue
            if name not in b.variables:
                report(f"Variable {name!r} missing from file 2")
                continue

            x, y = a[name], b[name]

            if x.dims != y.dims:
                report(f"{name}: dimensions {x.dims} vs {y.dims}")
                continue

            if x.shape != y.shape:
                report(f"{name}: shapes {x.shape} vs {y.shape}")
                continue

            if x.dtype != y.dtype:
                report(f"{name}: dtypes {x.dtype} vs {y.dtype}")

            compare_attrs(f"{name} attributes", x.attrs, y.attrs, report)

            # Compare actual values
            xv, yv = x.values, y.values

            if xv.dtype.kind in "iufc" and yv.dtype.kind in "iufc":
                if xv.dtype.kind == "c" or yv.dtype.kind == "c":
                    close = np.isclose(xv, yv, rtol=rtol, atol=atol, equal_nan=True)
                    delta = np.abs(xv - yv)
                else:
                    close = np.isclose(xv, yv, rtol=rtol, atol=atol, equal_nan=True)
                    x64 = xv.astype(np.float64)
                    y64 = yv.astype(np.float64)

                    delta = np.full_like(x64, np.inf)
                    finite = np.isfinite(x64) & np.isfinite(y64)
                    np.subtract(x64, y64, out=delta, where=finite)
                    np.abs(delta, out=delta)
                    delta[x64 == y64] = 0.0

                    rdelta = np.zeros_like(delta)
                    np.divide(
                        delta,
                        np.abs(y64),
                        out=rdelta,
                        where=(y64 != 0),
                    )
                    rdelta[(y64 == 0) & (delta != 0)] = np.inf

                if not np.all(close):
                    bad = ~close
                    report(
                        f"{name}: {bad.sum()} differing values; "
                        f"max abs diff = "
                        f"{np.max(delta[bad]):.6g}"
                        f" max rel diff = "
                        f"{np.max(rdelta[bad]):.6g}"
                    )
            elif not np.array_equal(xv, yv):
                report(f"{name}: values differ")

    print(
        f"\n{'Differences found: ' + str(differences) if differences else 'Files match.'}"
    )


def compare_attrs(label, a, b, report):
    for name in sorted(set(a) | set(b)):
        if name not in a:
            report(f"{label}: attribute {name!r} missing from file 1")
        elif name not in b:
            report(f"{label}: attribute {name!r} missing from file 2")
        else:
            x, y = np.asarray(a[name]), np.asarray(b[name])
            if x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
                report(f"{label}: attribute {name!r} differs")


if __name__ == "__main__":
    compare_netcdf(sys.argv[1], sys.argv[2])
