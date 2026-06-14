# Analysis report - 15min## Best model- model: single- window: 96- rmse_kwh: 8.1874## Top recomputed metricstrack  model  window  rmse_kwh  mae_kwh       r2      wia                                                                         model_path
15min single      96  8.187367 4.282085 0.932236 0.981801  /content/drive/MyDrive/Shared-Colab-Storage/Final/15min/single/models/win96.keras
15min double      16  8.374930 4.268899 0.928992 0.981279  /content/drive/MyDrive/Shared-Colab-Storage/Final/15min/double/models/win16.keras
15min  bidir      96  8.408039 4.407863 0.928534 0.980604   /content/drive/MyDrive/Shared-Colab-Storage/Final/15min/bidir/models/win96.keras
15min double      48  8.498960 4.670361 0.926915 0.980911  /content/drive/MyDrive/Shared-Colab-Storage/Final/15min/double/models/win48.keras
15min  bidir      48  8.566514 4.496929 0.925749 0.980415   /content/drive/MyDrive/Shared-Colab-Storage/Final/15min/bidir/models/win48.keras
15min  bidir     672  8.576240 4.503278 0.926302 0.980835  /content/drive/MyDrive/Shared-Colab-Storage/Final/15min/bidir/models/win672.keras
15min single       8  8.636859 4.411439 0.924475 0.980010   /content/drive/MyDrive/Shared-Colab-Storage/Final/15min/single/models/win8.keras
15min single      64  8.692376 4.472788 0.923572 0.980052  /content/drive/MyDrive/Shared-Colab-Storage/Final/15min/single/models/win64.keras
15min double     672  8.713934 4.707139 0.923916 0.980101 /content/drive/MyDrive/Shared-Colab-Storage/Final/15min/double/models/win672.keras
15min  bidir      16  8.720022 4.691804 0.923020 0.979150   /content/drive/MyDrive/Shared-Colab-Storage/Final/15min/bidir/models/win16.keras## SHAP/LIME/IG summarySee analysis/xai/, analysis/ig/, memory_erasure_*.csv, fidelity_*.csv, hidden_states/