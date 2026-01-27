======================================================================
BUILDING DATASET - Speaker ID + Embedding Verification
======================================================================

Collecting data...
📋 Speaker ID mapping:
   0: Adi
   1: Aleksander
   2: AlicjaMichal
   3: AnnaAleksander
   4: AnnaMichal
   5: Anne
   6: Bailey
   7: Buffet
   8: Churchill
   9: Dominika
   10: Emma
   11: FDR
   12: Greta
   13: GrianYT
   14: HomelessGuy
   15: IwoMichal
   16: JFK
   17: Julian
   18: JustExist
   19: Kaos
   20: Kevin
   21: KindCowboy
   22: KryptydaYT
   23: Lara
   24: LenaW
   25: Linus
   26: Malala
   27: Mantas
   28: Marzena
   29: Michal
   30: Natalia
   31: Nobel1
   32: Nobel4
   33: Obama
   34: OldMan
   35: Oppenheimer
   36: Oversimplified
   37: Pati
   38: Piotr
   39: Ponder
   40: Rafal
   41: Reagan
   42: Szyc
   43: Thatcher
   44: Theresa
   45: Torvalds
   46: WeronikaMichal
   47: chaplin
   48: gates
   49: jbp
   50: pacino
   51: qba
   52: queenElisabeth
   53: reeves
   54: smith
   55: sob
   56: trump
   57: turing
  ⚠️ Skipping Alexander-aleksander.mp3: not in labels.yaml
  ⚠️ Skipping Gallic-Wars-Aleksander.mp3: not in labels.yaml
  ⚠️ Skipping gatsby-ania.mp3: not in labels.yaml
  ⚠️ Skipping holmes-Lena.wav: not in labels.yaml
  ⚠️ Skipping invocation-Pan-Tadeusz.mp3: not in labels.yaml
  ⚠️ Skipping jfk-2-1.wav: not in labels.yaml
  ⚠️ Skipping Napoleon-aleksander.mp3: not in labels.yaml
  ⚠️ Skipping pati-glacier.mp3: not in labels.yaml
  ⚠️ Skipping prince-Aleksander.mp3: not in labels.yaml
  ⚠️ Skipping sartre-ania.wav: not in labels.yaml
  ⚠️ Skipping tiffany-Ania.wav: not in labels.yaml
  ⚠️ Skipping Bailey-1.mp3: not in labels.yaml
  ⚠️ Skipping Bailey-2.mp3: not in labels.yaml
  ⚠️ Skipping HomelessGuy-1.mp3: not in labels.yaml
  ⚠️ Skipping JustExist-1.mp3: not in labels.yaml
  ⚠️ Skipping Kaos-1.mp3: not in labels.yaml
  ⚠️ Skipping KindCowboy-1.mp3: not in labels.yaml
  ⚠️ Skipping KindCowboy-2.mp3: not in labels.yaml
  ⚠️ Skipping OldMan-1.mp3: not in labels.yaml
  ⚠️ Skipping OldMan-2.mp3: not in labels.yaml
  ⚠️ Skipping Pondering-1.mp3: not in labels.yaml
  ⚠️ Skipping Recording-1.m4a: not in labels.yaml
  ⚠️ Skipping Recording-2.m4a: not in labels.yaml
  ⚠️ Skipping Recording-3.m4a: not in labels.yaml
  ⚠️ Skipping Recording-4.m4a: not in labels.yaml
  ⚠️ Skipping anna-1.m4a: not in labels.yaml
  ⚠️ Skipping anna-2.m4a: not in labels.yaml
  ⚠️ Skipping anna-3.m4a: not in labels.yaml
  ⚠️ Skipping AUDIO-2025-10-29-19-59-12.m4a: not in labels.yaml
  ⚠️ Skipping AUDIO-2025-10-29-22-42-29.m4a: not in labels.yaml
  ⚠️ Skipping AUDIO-2025-10-29-22-42-30.m4a: not in labels.yaml
  ⚠️ Skipping Hemingway-Michał.m4a: not in labels.yaml
  ⚠️ Skipping iwo-iphone.m4a: not in labels.yaml
  ⚠️ Skipping iwo-laptop.m4a: not in labels.yaml
  ⚠️ Skipping iwo-słuchawki.m4a: not in labels.yaml
  ⚠️ Skipping michał-financial-times.m4a: not in labels.yaml
  ⚠️ Skipping Onet-michał.m4a: not in labels.yaml
  ⚠️ Skipping Warsaw-Central-3.m4a: not in labels.yaml
  ⚠️ Skipping Warsaw-Central-4.m4a: not in labels.yaml
  ⚠️ Skipping Warsaw-Central-5.m4a: not in labels.yaml
  ⚠️ Skipping Anne_F_False.m4a: not in labels.yaml
  ⚠️ Skipping Dominika_F_False.m4a: not in labels.yaml
  ⚠️ Skipping Emma_F_False.m4a: not in labels.yaml
  ⚠️ Skipping Greta1_F_False.m4a: not in labels.yaml
  ⚠️ Skipping Greta2_F_False.m4a: not in labels.yaml
  ⚠️ Skipping Greta3_F_False.m4a: not in labels.yaml
  ⚠️ Skipping Greta4_F_False.m4a: not in labels.yaml
  ⚠️ Skipping Julian_M_False.m4a: not in labels.yaml
  ⚠️ Skipping Lara_F_False.m4a: not in labels.yaml
  ⚠️ Skipping marzena-2.mp3: not in labels.yaml
  ⚠️ Skipping Marzena_F_False.m4a: not in labels.yaml
  ⚠️ Skipping Natalia_F_False.m4a: not in labels.yaml
  ⚠️ Skipping Obama2_M_False.m4a: not in labels.yaml
  ⚠️ Skipping Obama_M_False.m4a: not in labels.yaml
  ⚠️ Skipping Piotr_M_True.m4a: not in labels.yaml
  ⚠️ Skipping rafal_M_True.mp3: not in labels.yaml

📂 Found 107 audio files to process
👥 Unique speakers: 58
🔧 Using parallel processing with all CPU cores...
🚀 RUNNING AGGRESSIVE AUGMENTATION (High Overlap for Class 1)
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
[Parallel(n_jobs=-1)]: Done   1 tasks      | elapsed:   16.3s
[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed:   28.1s
[Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed:   40.4s
[Parallel(n_jobs=-1)]: Done  26 tasks      | elapsed:   56.5s
[Parallel(n_jobs=-1)]: Done  37 tasks      | elapsed:  1.4min
[Parallel(n_jobs=-1)]: Done  48 tasks      | elapsed:  1.7min
[Parallel(n_jobs=-1)]: Done  61 tasks      | elapsed:  2.3min
[Parallel(n_jobs=-1)]: Done  74 tasks      | elapsed:  2.6min
[Parallel(n_jobs=-1)]: Done  95 out of 107 | elapsed:  3.1min remaining:   23.4s
[Parallel(n_jobs=-1)]: Done 107 out of 107 | elapsed:  5.7min finished
✅ Processed 250066 total chunks from 107 files

📊 Speaker distribution (labels are speaker IDs):
   🔴 OUT ID 0: Adi                  - 1915 chunks
   🟢 IN ID 1: Aleksander           - 22550 chunks
   🔴 OUT ID 2: AlicjaMichal         - 1855 chunks
   🔴 OUT ID 3: AnnaAleksander       - 3870 chunks
   🔴 OUT ID 4: AnnaMichal           - 1695 chunks
   🔴 OUT ID 5: Anne                 - 1300 chunks
   🔴 OUT ID 6: Bailey               - 5690 chunks
   🔴 OUT ID 7: Buffet               - 3435 chunks
   🔴 OUT ID 8: Churchill            - 1360 chunks
   🔴 OUT ID 9: Dominika             - 1525 chunks
   🔴 OUT ID 10: Emma                 - 1300 chunks
   🔴 OUT ID 11: FDR                  - 1435 chunks
   🔴 OUT ID 12: Greta                - 4755 chunks
   🔴 OUT ID 13: GrianYT              - 1730 chunks
   🔴 OUT ID 14: HomelessGuy          - 1815 chunks
   🔴 OUT ID 15: IwoMichal            - 2055 chunks
   🔴 OUT ID 16: JFK                  - 4215 chunks
   🔴 OUT ID 17: Julian               - 1525 chunks
   🔴 OUT ID 18: JustExist            - 2890 chunks
   🔴 OUT ID 19: Kaos                 - 635 chunks
   🔴 OUT ID 20: Kevin                - 355 chunks
   🔴 OUT ID 21: KindCowboy           - 11885 chunks
   🔴 OUT ID 22: KryptydaYT           - 2040 chunks
   🔴 OUT ID 23: Lara                 - 1260 chunks
   🔴 OUT ID 24: LenaW                - 1595 chunks
   🔴 OUT ID 25: Linus                - 3770 chunks
   🔴 OUT ID 26: Malala               - 1345 chunks
   🟢 IN ID 27: Mantas               - 38380 chunks
   🔴 OUT ID 28: Marzena              - 1870 chunks
   🟢 IN ID 29: Michal               - 23417 chunks
   🔴 OUT ID 30: Natalia              - 1215 chunks
   🔴 OUT ID 31: Nobel1               - 935 chunks
   🔴 OUT ID 32: Nobel4               - 795 chunks
   🔴 OUT ID 33: Obama                - 2585 chunks
   🔴 OUT ID 34: OldMan               - 4860 chunks
   🔴 OUT ID 35: Oppenheimer          - 280 chunks
   🔴 OUT ID 36: Oversimplified       - 1635 chunks
   🔴 OUT ID 37: Pati                 - 1230 chunks
   🟢 IN ID 38: Piotr                - 33990 chunks
   🔴 OUT ID 39: Ponder               - 4710 chunks
   🟢 IN ID 40: Rafal                - 5344 chunks
   🔴 OUT ID 41: Reagan               - 2645 chunks
   🔴 OUT ID 42: Szyc                 - 360 chunks
   🔴 OUT ID 43: Thatcher             - 1255 chunks
   🔴 OUT ID 44: Theresa              - 730 chunks
   🔴 OUT ID 45: Torvalds             - 1025 chunks
   🔴 OUT ID 46: WeronikaMichal       - 2000 chunks
   🔴 OUT ID 47: chaplin              - 1020 chunks
   🔴 OUT ID 48: gates                - 1490 chunks
   🔴 OUT ID 49: jbp                  - 2515 chunks
   🔴 OUT ID 50: pacino               - 2075 chunks
   🔴 OUT ID 51: qba                  - 3300 chunks
   🔴 OUT ID 52: queenElisabeth       - 5925 chunks
   🔴 OUT ID 53: reeves               - 3305 chunks
   🔴 OUT ID 54: smith                - 1205 chunks
   🔴 OUT ID 55: sob                  - 7535 chunks
   🔴 OUT ID 56: trump                - 2355 chunks
   🔴 OUT ID 57: turing               - 280 chunks

Total chunks: 250066
Unique speakers: 58

Building splits (80/10/10 per speaker)...

📋 SPEAKER-LEVEL SPLIT STRATEGY (Approach 2: Embedding-based):
   Total speakers: 58

   🟢 IN-GROUP speakers (access granted):
      ID 1: Aleksander           - 22550 chunks
      ID 27: Mantas               - 38380 chunks
      ID 29: Michal               - 23417 chunks
      ID 38: Piotr                - 33990 chunks
      ID 40: Rafal                - 5344 chunks

   🔴 OUT-GROUP speakers (access denied):
      ID 0: Adi                  - 1915 chunks
      ID 2: AlicjaMichal         - 1855 chunks
      ID 3: AnnaAleksander       - 3870 chunks
      ID 4: AnnaMichal           - 1695 chunks
      ID 5: Anne                 - 1300 chunks
      ID 6: Bailey               - 5690 chunks
      ID 7: Buffet               - 3435 chunks
      ID 8: Churchill            - 1360 chunks
      ID 9: Dominika             - 1525 chunks
      ID 10: Emma                 - 1300 chunks
      ID 11: FDR                  - 1435 chunks
      ID 12: Greta                - 4755 chunks
      ID 13: GrianYT              - 1730 chunks
      ID 14: HomelessGuy          - 1815 chunks
      ID 15: IwoMichal            - 2055 chunks
      ID 16: JFK                  - 4215 chunks
      ID 17: Julian               - 1525 chunks
      ID 18: JustExist            - 2890 chunks
      ID 19: Kaos                 - 635 chunks
      ID 20: Kevin                - 355 chunks
      ID 21: KindCowboy           - 11885 chunks
      ID 22: KryptydaYT           - 2040 chunks
      ID 23: Lara                 - 1260 chunks
      ID 24: LenaW                - 1595 chunks
      ID 25: Linus                - 3770 chunks
      ID 26: Malala               - 1345 chunks
      ID 28: Marzena              - 1870 chunks
      ID 30: Natalia              - 1215 chunks
      ID 31: Nobel1               - 935 chunks
      ID 32: Nobel4               - 795 chunks
      ID 33: Obama                - 2585 chunks
      ID 34: OldMan               - 4860 chunks
      ID 35: Oppenheimer          - 280 chunks
      ID 36: Oversimplified       - 1635 chunks
      ID 37: Pati                 - 1230 chunks
      ID 39: Ponder               - 4710 chunks
      ID 41: Reagan               - 2645 chunks
      ID 42: Szyc                 - 360 chunks
      ID 43: Thatcher             - 1255 chunks
      ID 44: Theresa              - 730 chunks
      ID 45: Torvalds             - 1025 chunks
      ID 46: WeronikaMichal       - 2000 chunks
      ID 47: chaplin              - 1020 chunks
      ID 48: gates                - 1490 chunks
      ID 49: jbp                  - 2515 chunks
      ID 50: pacino               - 2075 chunks
      ID 51: qba                  - 3300 chunks
      ID 52: queenElisabeth       - 5925 chunks
      ID 53: reeves               - 3305 chunks
      ID 54: smith                - 1205 chunks
      ID 55: sob                  - 7535 chunks
      ID 56: trump                - 2355 chunks
      ID 57: turing               - 280 chunks

   Allocating all speakers (80/10/10 per speaker):
      [✗] ID 0 Adi                 : 1532 train, 191 val, 192 test
      [✓] ID 1 Aleksander          : 18040 train, 2255 val, 2255 test
      [✗] ID 2 AlicjaMichal        : 1484 train, 185 val, 186 test
      [✗] ID 3 AnnaAleksander      : 3096 train, 387 val, 387 test
      [✗] ID 4 AnnaMichal          : 1356 train, 169 val, 170 test
      [✗] ID 5 Anne                : 1040 train, 130 val, 130 test
      [✗] ID 6 Bailey              : 4552 train, 569 val, 569 test
      [✗] ID 7 Buffet              : 2748 train, 343 val, 344 test
      [✗] ID 8 Churchill           : 1088 train, 136 val, 136 test
      [✗] ID 9 Dominika            : 1220 train, 152 val, 153 test
      [✗] ID 10 Emma                : 1040 train, 130 val, 130 test
      [✗] ID 11 FDR                 : 1148 train, 143 val, 144 test
      [✗] ID 12 Greta               : 3804 train, 475 val, 476 test
      [✗] ID 13 GrianYT             : 1384 train, 173 val, 173 test
      [✗] ID 14 HomelessGuy         : 1452 train, 181 val, 182 test
      [✗] ID 15 IwoMichal           : 1644 train, 205 val, 206 test
      [✗] ID 16 JFK                 : 3372 train, 421 val, 422 test
      [✗] ID 17 Julian              : 1220 train, 152 val, 153 test
      [✗] ID 18 JustExist           : 2312 train, 289 val, 289 test
      [✗] ID 19 Kaos                : 508 train, 63 val, 64 test
      [✗] ID 20 Kevin               : 284 train, 35 val, 36 test
      [✗] ID 21 KindCowboy          : 9508 train, 1188 val, 1189 test
      [✗] ID 22 KryptydaYT          : 1632 train, 204 val, 204 test
      [✗] ID 23 Lara                : 1008 train, 126 val, 126 test
      [✗] ID 24 LenaW               : 1276 train, 159 val, 160 test
      [✗] ID 25 Linus               : 3016 train, 377 val, 377 test
      [✗] ID 26 Malala              : 1076 train, 134 val, 135 test
      [✓] ID 27 Mantas              : 30704 train, 3838 val, 3838 test
      [✗] ID 28 Marzena             : 1496 train, 187 val, 187 test
      [✓] ID 29 Michal              : 18733 train, 2341 val, 2343 test
      [✗] ID 30 Natalia             : 972 train, 121 val, 122 test
      [✗] ID 31 Nobel1              : 748 train, 93 val, 94 test
      [✗] ID 32 Nobel4              : 636 train, 79 val, 80 test
      [✗] ID 33 Obama               : 2068 train, 258 val, 259 test
      [✗] ID 34 OldMan              : 3888 train, 486 val, 486 test
      [✗] ID 35 Oppenheimer         : 224 train, 28 val, 28 test
      [✗] ID 36 Oversimplified      : 1308 train, 163 val, 164 test
      [✗] ID 37 Pati                : 984 train, 123 val, 123 test
      [✓] ID 38 Piotr               : 27192 train, 3399 val, 3399 test
      [✗] ID 39 Ponder              : 3768 train, 471 val, 471 test
      [✓] ID 40 Rafal               : 4275 train, 534 val, 535 test
      [✗] ID 41 Reagan              : 2116 train, 264 val, 265 test
      [✗] ID 42 Szyc                : 288 train, 36 val, 36 test
      [✗] ID 43 Thatcher            : 1004 train, 125 val, 126 test
      [✗] ID 44 Theresa             : 584 train, 73 val, 73 test
      [✗] ID 45 Torvalds            : 820 train, 102 val, 103 test
      [✗] ID 46 WeronikaMichal      : 1600 train, 200 val, 200 test
      [✗] ID 47 chaplin             : 816 train, 102 val, 102 test
      [✗] ID 48 gates               : 1192 train, 149 val, 149 test
      [✗] ID 49 jbp                 : 2012 train, 251 val, 252 test
      [✗] ID 50 pacino              : 1660 train, 207 val, 208 test
      [✗] ID 51 qba                 : 2640 train, 330 val, 330 test
      [✗] ID 52 queenElisabeth      : 4740 train, 592 val, 593 test
      [✗] ID 53 reeves              : 2644 train, 330 val, 331 test
      [✗] ID 54 smith               : 964 train, 120 val, 121 test
      [✗] ID 55 sob                 : 6028 train, 753 val, 754 test
      [✗] ID 56 trump               : 1884 train, 235 val, 236 test
      [✗] ID 57 turing              : 224 train, 28 val, 28 test

📊 CHUNK-LEVEL SPLIT SUMMARY:
   train: 200052 chunks | 58 speakers | In-group: 98944 | Out-group: 101108
   val  :  24990 chunks | 58 speakers | In-group: 12367 | Out-group: 12623
   test :  25024 chunks | 58 speakers | In-group: 12370 | Out-group: 12654

   ✓ LABEL_MODE = 'speaker_id'
   ✓ Labels are SPEAKER IDs (multi-class: 0 to 57)
   ✓ in_group metadata stored for inference-time access control
   ✓ Each speaker has chunks in training set

Saving HDF5...
✅ Saved HDF5 → c:\Users\pczec\Desktop\Studia\SEM5\IML\IML-PW\outputs\logmels_spkid_aug_26-01-27_16-06-04.h5
   Mode: speaker_id
   Task: speaker_verification (embedding + threshold)
   Speakers: 58 total (5 in-group, 53 out-group)

======================================================================
✅ Done in 491.0s
📁 Saved to: c:\Users\pczec\Desktop\Studia\SEM5\IML\IML-PW\outputs\logmels_spkid_aug_26-01-27_16-06-04.h5
======================================================================

📋 Final split statistics:
   train: 200052 samples | 58 speakers | In-group: 98944 | Out-group: 101108
   val: 24990 samples | 58 speakers | In-group: 12367 | Out-group: 12623
   test: 25024 samples | 58 speakers | In-group: 12370 | Out-group: 12654