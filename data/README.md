# Local data layout

The repository does not redistribute datasets. Create this local structure:

```text
data/
├── official/
│   └── data.csv
├── curated/
│   └── contrast.csv       # optional; same five columns
└── derived/               # generated and ignored by Git
```

Required CSV columns are `id,name,description,category,label`. Categories must
be exactly `БАД` or `Легковоспламеняющиеся`; labels are integers 0 or 1.

To train the historical lineage, use a lawfully obtained frozen
`lora_dataset_plus.jsonl` whose expected hash appears in the manifest. Do not
substitute a later mutable export while keeping the historical release name.

