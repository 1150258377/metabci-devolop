import mat73, scipy.io, sys, os, json, h5py
mat_path='DREAMER.mat'
try:
    d=mat73.loadmat(mat_path)
    print('mat73 keys', list(d.keys()))
    print('type', type(next(iter(d.values()))))
    print('DREAMER type', type(d['DREAMER']))
    print('DREAMER shape', getattr(d['DREAMER'], 'shape', None))
    try:
        first = d['DREAMER'][0]
        print('first element type', type(first))
        print('first element keys', [name for name in dir(first) if not name.startswith('_')][:10])
    except Exception as exc:
        print('indexing error', exc)
except Exception as e:
    print('mat73 failed', e)
    d=scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    print('scipy keys', list(d.keys()))
    print('DREAMER type', type(d['DREAMER']))
    import numpy as np
    print('DREAMER shape', getattr(d['DREAMER'], 'shape', None))
    try:
        first = d['DREAMER'][0]
        print('first element type', type(first))
        # print structure fields if recarray or np.void
        if hasattr(first, '_fieldnames'):
            print('fieldnames', first._fieldnames)
        print('fields', [k for k in first.__dict__.keys()])
    except Exception as exc:
        print('indexing error', exc)

struct = d['DREAMER']
print('fieldnames', getattr(struct, '_fieldnames', None))

data_struct=struct.Data
print('Data type', type(data_struct))
try:
    print('Data length', len(data_struct))
    print('First element type', type(data_struct[0]))
    first_subj = data_struct[0]
    print('first_subj fieldnames', getattr(first_subj, '_fieldnames', None))
    eeg = first_subj.EEG
    print('EEG shape', getattr(eeg, 'shape', None), type(eeg))
    try:
        print('EEG length', len(eeg))
        print('first video type', type(eeg[0]))
    except Exception as e3:
        print('EEG cannot iterate', e3)
    print('EEG fieldnames', getattr(eeg, '_fieldnames', None))
    stim = eeg.stimuli
    print('stim type', type(stim))
    print('stim length', len(stim))
    print('first stim shape', getattr(stim[0], 'shape', None))
    print('ScoreValence', first_subj.ScoreValence.shape)
except Exception as e2:
    print('cannot iterate', e2) 