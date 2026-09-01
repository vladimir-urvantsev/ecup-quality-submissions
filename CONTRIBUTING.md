# Contributing

Changes to the verdict path require a paired evaluation against the frozen
reference. Changes to rationale text must preserve the already selected binary
verdict and pass the wire-format tests. Do not commit product data, images,
predictions, model weights, credentials or machine-specific paths.

Before proposing a change, run:

```bash
make test
make audit
```

