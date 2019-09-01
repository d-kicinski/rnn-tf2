
# RNN - benchmark
```text
Running DynamicLSTM
batch_size: 10	 sequence_length: 10	 use_sequence_length_info=False	  TIME: 0.252104 [s]
batch_size: 10	 sequence_length: 50	 use_sequence_length_info=False	  TIME: 0.647838 [s]
batch_size: 10	 sequence_length: 200	 use_sequence_length_info=False	  TIME: 2.3494 [s]
batch_size: 10	 sequence_length: 500	 use_sequence_length_info=False	  TIME: 5.64942 [s]
batch_size: 10	 sequence_length: 1000	 use_sequence_length_info=False	  TIME: 11.2493 [s]
batch_size: 10	 sequence_length: 1000	 use_sequence_length_info=False	  TIME: 11.4055 [s]
batch_size: 10	 sequence_length: 500	 use_sequence_length_info=False	  TIME: 5.79567 [s]
batch_size: 10	 sequence_length: 200	 use_sequence_length_info=False	  TIME: 2.62515 [s]
batch_size: 10	 sequence_length: 50	 use_sequence_length_info=False	  TIME: 0.775688 [s]
batch_size: 10	 sequence_length: 10	 use_sequence_length_info=False	  TIME: 0.2175 [s]
TOTAL TIME: 40.9681 [s] 

Running DynamicLSTM
batch_size: 10	 sequence_length: 10	 use_sequence_length_info=True	  TIME: 0.254788 [s]
batch_size: 10	 sequence_length: 50	 use_sequence_length_info=True	  TIME: 0.643708 [s]
batch_size: 10	 sequence_length: 200	 use_sequence_length_info=True	  TIME: 2.2699 [s]
batch_size: 10	 sequence_length: 500	 use_sequence_length_info=True	  TIME: 5.12085 [s]
batch_size: 10	 sequence_length: 1000	 use_sequence_length_info=True	  TIME: 9.60411 [s]
batch_size: 10	 sequence_length: 1000	 use_sequence_length_info=True	  TIME: 9.64292 [s]
batch_size: 10	 sequence_length: 500	 use_sequence_length_info=True	  TIME: 4.81813 [s]
batch_size: 10	 sequence_length: 200	 use_sequence_length_info=True	  TIME: 1.8835 [s]
batch_size: 10	 sequence_length: 50	 use_sequence_length_info=True	  TIME: 0.576198 [s]
batch_size: 10	 sequence_length: 10	 use_sequence_length_info=True	  TIME: 0.194758 [s]
TOTAL TIME: 35.0094 [s]
```

```text
Running LiteDynamicLSTM
batch_size: 10	 sequence_length: 10	 use_sequence_length_info=False	  TIME: 0.328576 [s]
batch_size: 10	 sequence_length: 50	 use_sequence_length_info=False	  TIME: 1.03419 [s]
batch_size: 10	 sequence_length: 200	 use_sequence_length_info=False	  TIME: 3.9764 [s]
batch_size: 10	 sequence_length: 500	 use_sequence_length_info=False	  TIME: 9.87624 [s]
batch_size: 10	 sequence_length: 1000	 use_sequence_length_info=False	  TIME: 20.1918 [s]
batch_size: 10	 sequence_length: 1000	 use_sequence_length_info=False	  TIME: 20.1761 [s]
batch_size: 10	 sequence_length: 500	 use_sequence_length_info=False	  TIME: 9.98817 [s]
batch_size: 10	 sequence_length: 200	 use_sequence_length_info=False	  TIME: 4.04245 [s]
batch_size: 10	 sequence_length: 50	 use_sequence_length_info=False	  TIME: 1.02816 [s]
batch_size: 10	 sequence_length: 10	 use_sequence_length_info=False	  TIME: 0.268831 [s]
TOTAL TIME: 70.9115 [s]

Running LiteDynamicLSTM
batch_size: 10	 sequence_length: 10	 use_sequence_length_info=True	  TIME: 0.328085 [s]
batch_size: 10	 sequence_length: 50	 use_sequence_length_info=True	  TIME: 0.993531 [s]
batch_size: 10	 sequence_length: 200	 use_sequence_length_info=True	  TIME: 4.08191 [s]
batch_size: 10	 sequence_length: 500	 use_sequence_length_info=True	  TIME: 10.5202 [s]
batch_size: 10	 sequence_length: 1000	 use_sequence_length_info=True	  TIME: 19.922 [s]
batch_size: 10	 sequence_length: 1000	 use_sequence_length_info=True	  TIME: 19.5463 [s]
batch_size: 10	 sequence_length: 500	 use_sequence_length_info=True	  TIME: 9.71969 [s]
batch_size: 10	 sequence_length: 200	 use_sequence_length_info=True	  TIME: 3.89329 [s]
batch_size: 10	 sequence_length: 50	 use_sequence_length_info=True	  TIME: 0.982882 [s]
batch_size: 10	 sequence_length: 10	 use_sequence_length_info=True	  TIME: 0.276426 [s]
TOTAL TIME: 70.2648 [s]
```

```text
Running StaticLSTM
batch_size: 10	 sequence_length: 10	 use_sequence_length_info=False	  TIME: 13.6136 [s]
batch_size: 10	 sequence_length: 50	 use_sequence_length_info=False	  TIME: 10.4408 [s]
batch_size: 10	 sequence_length: 200	 use_sequence_length_info=False	  TIME: 11.3706 [s]
batch_size: 10	 sequence_length: 500	 use_sequence_length_info=False	  TIME: 11.7504 [s]
batch_size: 10	 sequence_length: 1000	 use_sequence_length_info=False	  TIME: 11.581 [s]
batch_size: 10	 sequence_length: 1000	 use_sequence_length_info=False	  TIME: 10.7659 [s]
batch_size: 10	 sequence_length: 500	 use_sequence_length_info=False	  TIME: 12.0853 [s]
batch_size: 10	 sequence_length: 200	 use_sequence_length_info=False	  TIME: 11.6063 [s]
batch_size: 10	 sequence_length: 50	 use_sequence_length_info=False	  TIME: 11.5367 [s]
batch_size: 10	 sequence_length: 10	 use_sequence_length_info=False	  TIME: 11.7197 [s]
TOTAL TIME: 116.471 [s]

batch_size: 10	 sequence_length: 10	 use_sequence_length_info=True	  TIME: 50.192 [s]
batch_size: 10	 sequence_length: 50	 use_sequence_length_info=True	  TIME: 5.64104 [s]
batch_size: 10	 sequence_length: 200	 use_sequence_length_info=True	  TIME: 7.0056 [s]
batch_size: 10	 sequence_length: 500	 use_sequence_length_info=True	  TIME: 9.70772 [s]
batch_size: 10	 sequence_length: 1000	 use_sequence_length_info=True	  TIME: 14.3721 [s]
batch_size: 10	 sequence_length: 1000	 use_sequence_length_info=True	  TIME: 14.7313 [s]
batch_size: 10	 sequence_length: 500	 use_sequence_length_info=True	  TIME: 9.93148 [s]
batch_size: 10	 sequence_length: 200	 use_sequence_length_info=True	  TIME: 6.98246 [s]
batch_size: 10	 sequence_length: 50	 use_sequence_length_info=True	  TIME: 5.37093 [s]
batch_size: 10	 sequence_length: 10	 use_sequence_length_info=True	  TIME: 4.87672 [s]
TOTAL TIME: 128.812 [s]
```


Remarks:
- for smaller max_sequence_length (eg. 500 instead of 1000) static_rnn seems to be faster than LiteDynamicLSTM
