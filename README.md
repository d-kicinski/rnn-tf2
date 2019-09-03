
# RNN - benchmark

 Short(ish) sequences
 
| model            | use_sequence_length_info | time[s] |
|------------------|--------------------------|---------|
| LiteDynamicLSTM  | false                    | 147.823 |
| LiteDynamicLSTM  | true                     | 140.532 |
| DynamicLSTM      | false                    | 118.088 |
| DynamicLSTM      | true                     | 114.407 |
| StaticLSTM       | false                    | 274.508 |
| StaticLSTM       | true                     | 149.978 |
| KerasDynamicLSTM | N/A                      | 129.592 |
| KerasStaticLSTM  | N/A                      | 132.017 |

Long sequences

| model            | use_sequence_length_info | time[s] |
|------------------|--------------------------|---------|
| LiteDynamicLSTM  | false                    | 1011.18 |
| LiteDynamicLSTM  | true                     | 946.535 |
| DynamicLSTM      | false                    | 797.156 |
| DynamicLSTM      | true                     | 760.779 |
| StaticLSTM       | false                    | N/A     |
| StaticLSTM       | true                     | N/A     |
| KerasDynamicLSTM | N/A                      | 791.499 |
| KerasStaticLSTM  | N/A                      | 940.599 |

See below for more detailed results


## Short(ish) sequences
### LiteDynamicLSTM
```text
Running LiteDynamicLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=False	  TIME: 3.05065 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=False	  TIME: 13.9029 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=False	  TIME: 57.1308 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=False	  TIME: 56.8976 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=False	  TIME: 13.8953 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=False	  TIME: 2.94496 [s]
TOTAL TIME: 147.823 [s]

Running LiteDynamicLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=True	  TIME: 3.03282 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=True	  TIME: 13.5491 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=True	  TIME: 53.7079 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=True	  TIME: 53.6887 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=True	  TIME: 13.5898 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=True	  TIME: 2.96305 [s]
TOTAL TIME: 140.532 [s]
```

### DynamicLSTM
```text
Running DynamicLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=False	  TIME: 2.65967 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=False	  TIME: 11.4257 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=False	  TIME: 44.9419 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=False	  TIME: 45.0146 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=False	  TIME: 11.395 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=False	  TIME: 2.65116 [s]
TOTAL TIME: 118.088 [s]

Running DynamicLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=True	  TIME: 2.64009 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=True	  TIME: 11.2485 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=True	  TIME: 43.2503 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=True	  TIME: 43.3949 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=True	  TIME: 11.2595 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=True	  TIME: 2.61345 [s]
TOTAL TIME: 114.407 [s]
```

### StaticLSTM
```text
Running StaticLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=False	  TIME: 46.1981 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=False	  TIME: 45.7942 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=False	  TIME: 45.9468 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=False	  TIME: 45.7277 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=False	  TIME: 45.389 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=False	  TIME: 45.4515 [s]
TOTAL TIME: 274.508 [s]

Running StaticLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=True	  TIME: 12.4991 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=True	  TIME: 16.2948 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=True	  TIME: 49.0124 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=True	  TIME: 48.6762 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=True	  TIME: 16.5695 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=True	  TIME: 6.92596 [s]
TOTAL TIME: 149.978 [s]
```

### KerasDynamicLSTM
```text
Running KerasDynamicLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 10.0	  TIME: 3.00792 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 50.0	  TIME: 11.4033 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 200.0	  TIME: 46.9786 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 200.0	  TIME: 52.396 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 50.0	  TIME: 12.6862 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 10.0	  TIME: 3.11999 [s]
TOTAL TIME: 129.592 [s]

```

### KerasStaticLSTM
```text
Running KerasStaticLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 10.0	  TIME: 3.27552 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 50.0	  TIME: 12.9252 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 200.0	  TIME: 52.8201 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 200.0	  TIME: 48.0225 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 50.0	  TIME: 12.032 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 10.0	  TIME: 2.94103 [s]
TOTAL TIME: 132.017 [s]
```

## Long sequences

### LiteDynamicLSTM
```text
Running LiteDynamicLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=False	  TIME: 3.03004 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=False	  TIME: 13.8772 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=False	  TIME: 56.8612 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 500.0	 use_sequence_length_info=False	  TIME: 143.55 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 1000.0	 use_sequence_length_info=False	  TIME: 288.403 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 1000.0	 use_sequence_length_info=False	  TIME: 288.124 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 500.0	 use_sequence_length_info=False	  TIME: 143.543 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=False	  TIME: 56.9519 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=False	  TIME: 13.8722 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=False	  TIME: 2.96572 [s]
TOTAL TIME: 1011.18 [s]

Running LiteDynamicLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=True	  TIME: 3.07336 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=True	  TIME: 13.6049 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=True	  TIME: 53.7864 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 500.0	 use_sequence_length_info=True	  TIME: 134.396 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 1000.0	 use_sequence_length_info=True	  TIME: 268.729 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 1000.0	 use_sequence_length_info=True	  TIME: 268.745 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 500.0	 use_sequence_length_info=True	  TIME: 134.066 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=True	  TIME: 53.6231 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=True	  TIME: 13.5343 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=True	  TIME: 2.97583 [s]
TOTAL TIME: 946.535 [s]
```

### DynamicLSTM
```text
Running DynamicLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=False	  TIME: 2.70162 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=False	  TIME: 11.4078 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=False	  TIME: 44.9585 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 500.0	 use_sequence_length_info=False	  TIME: 112.634 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 1000.0	 use_sequence_length_info=False	  TIME: 226.873 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 1000.0	 use_sequence_length_info=False	  TIME: 226.747 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 500.0	 use_sequence_length_info=False	  TIME: 112.955 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=False	  TIME: 44.9215 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=False	  TIME: 11.3865 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=False	  TIME: 2.56997 [s]
TOTAL TIME: 797.156 [s]

Running DynamicLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=True	  TIME: 2.72261 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=True	  TIME: 11.2429 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=True	  TIME: 43.1247 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 500.0	 use_sequence_length_info=True	  TIME: 107.337 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 1000.0	 use_sequence_length_info=True	  TIME: 215.85 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 1000.0	 use_sequence_length_info=True	  TIME: 216.289 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 500.0	 use_sequence_length_info=True	  TIME: 107.229 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 200.0	 use_sequence_length_info=True	  TIME: 43.1332 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 50.0	 use_sequence_length_info=True	  TIME: 11.2691 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	 sequence_length: 10.0	 use_sequence_length_info=True	  TIME: 2.58165 [s]
TOTAL TIME: 760.779 [s]
```

### StaticLSTM
```text
N/A
```

### KerasDynamicLSTM
```text
Running KerasDynamicLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 10.0	  TIME: 3.30141 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 50.0	  TIME: 12.9111 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 200.0	  TIME: 50.1907 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 500.0	  TIME: 122.757 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 1000.0	  TIME: 233.756 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 1000.0	  TIME: 210.173 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 500.0	  TIME: 103.22 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 200.0	  TIME: 41.5255 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 50.0	  TIME: 11.1401 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 10.0	  TIME: 2.52222 [s]
TOTAL TIME: 791.499 [s]
```

### KerasStaticLSTM
```text
Running KerasStaticLSTM
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 10.0	  TIME: 2.8986 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 50.0	  TIME: 12.0052 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 200.0	  TIME: 52.2024 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 500.0	  TIME: 138.224 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 1000.0	  TIME: 282.209 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 1000.0	  TIME: 259.521 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 500.0	  TIME: 130.111 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 200.0	  TIME: 49.4988 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 50.0	  TIME: 11.3594 [s]
epoch_count: 10	 batch_count: 50	 batch_size: 32	  sequence_length: 10.0	  TIME: 2.56909 [s]
TOTAL TIME: 940.599 [s]
```

