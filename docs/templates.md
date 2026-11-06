# Templates

In order for SIREN to run as quickly as possible, it bundles all its templates into a large matrix (prebuild_templates.jld2) which Julia can handle rapidly. The full list of templates that SIREN uses can still be accessed as the individual plain text files in the SIREN folder. 
```bash
cd templates/transient_template_dir
 ```
 or 
 ```bash
cd templates/host_template_dir
 ```
 The current template types available currently and number of them is displayed below:

| Transient Type | Template Count |
| :--- | :--- |
| `Ia` | 224 |
| `Ia-02cx` | 23 |
| `Ia-02es` | 16 |
| `Ia-91bg` | 44 |
| `Ia-91t` | 61 |
| `Ia-99aa` | 14 |
| `Ia-Ca-rich` | 18 |
| `Ia-CSM` | 24 |
| `Ia-CSM-(ambigious)` | 9 |
| `Ia-pec` | 52 |
| `Ia-rapid` | 4 |
| `PISN` | 9 |
| `FBOT` | 3 |
| `Ib` | 63 |
| `Ib_c` | 4 |
| `Ib-Ca-rich` | 21 |
| `Ib-norm` | 15 |
| `Ib-pec` | 5 |
| `Ibn` | 42 |
| `Ic` | 47 |
| `Ic-BL` | 50 |
| `Ic-broad` | 21 |
| `Ic-Ca-rich` | 4 |
| `Ic-norm` | 51 |
| `Ic-pec` | 8 |
| `II` | 127 |
| `II-flash` | 22 |
| `II-pec` | 6 |
| `IIb` | 138 |
| `IIb-flash` | 3 |
| `IIL` | 11 |
| `IIn` | 101 |
| `IIn-pec` | 6 |
| `IIP` | 259 |
| `ILRT` | 13 |
| `LBV` | 3 |
| `SLSN-I` | 135 |
| `SLSN-Ib` | 8 |
| `SLSN-II` | 24 |
| `SLSN-IIb` | 2 |
| `SLSN-IIn` | 20 |
| `SN-Imposter` | 6 |
| `super-chandra` | 37 |
| `TDE` | 15 |
| `TDE-Featless` | 10 |
| `TDE-H` | 24 |
| `TDE-H-He` | 46 |
| `TDE-He` | 2 |

| Galaxy Classification | Host Template File(s) |
| :--- | :--- |
| Elliptical | `E.txt` |
| Lenticular (S0) | `S0.txt` |
| Spiral Galaxies (Sa, Sb, Sc) | `Sa.txt`, `Sb.txt`, `Sc.txt` |
| Starburst Galaxies | `SB1.txt`, `SB2.txt`, `SB3.txt`, `SB4.txt`, `SB5.txt`, `SB6.txt` |