# TM Neural Map Recognizer

A neural network that decides whether a Terra Mystica map was created by the
[Terra Mystica Map Generator](https://github.com/ikmMaierBTUCS/Terra-Mystica-Map-Generator) ([version 4](https://github.com/ikmMaierBTUCS/Terra-Mystica-Map-Generator/tree/v4.0)),
which builds balanced maps, or by a deliberately "dumb" generator based on very simple rules.

## Architecture

We use a **GCN (Graph Convolutional Network)**. A hex map is naturally a graph: hexes are nodes, neighbourhood
relations are edges. A study on Catan boards found GCNs better than CNNs at capturing the board structure
([IPSJ](https://ipsj.ixsq.nii.ac.jp/records/240732)), and our own experiments confirmed this for Terra Mystica maps.
<!-- (GCN 0.986 vs. hexagonal CNN 0.963 vs. plain CNN 0.963 vs. fully connected MLP 0.73 validation accuracy).
GCNs were introduced by Kipf & Welling (*Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017);
our graph layer has separate weights for a hex itself and for its neighbours ("GraphConv", Morris et al.,
*Weisfeiler and Leman Go Neural*, AAAI 2019).-->

The 8 landscape types (7 terrain colours + river) are represented one-hot, i.e. 8-dimensionally.
It surprized me that a circular representation of the colour wheel as (sin(2πk/7), cos(2πk/7)) turned out to be disadvantageous.
<!--an error analysis showed that it systematically misjudges blue, while one-hot shows no measurable systematic error.-->
Plain one-hot encoding has also been used successfully for other circular data such as numbers modulo p
(Power et al., *Grokking*, 2022; Nanda et al., *Progress measures for grokking via mechanistic interpretability*,
ICLR 2023). 
<!--Nanda et al. show that the network learns cos/sin representations from one-hot input by itself, for
several frequencies, whereas our hand-made circular encoding contained only one.-->

A map with 113 hexes (land and river) is therefore represented as a 113 × 8 dimensional input.

**First hidden layer (graph layer).** Through its design I point the network to neighbourhood. Each hex gets
16 neurons, so 2 values per landscape type can be stored. Each of these neurons has only 8 + 8 inputs: 8 from the input values of its own hex and 8 from the
**sum** of the input values of its 6 neighbours (hexes at the border have only 2 to 5 neighbours):

    h_v[k] = ReLU( a_k · x_v  +  b_k · Σ_{u neighbour of v} x_u  +  c_k ),   k = 1 … 16

Here x_v is the 8-dimensional one-hot vector of hex v, so Σ x_u counts how many neighbours have each landscape type.
For each neuron k, a_k and b_k are 8-dimensional weight vectors and c_k is a single number, the bias.
<!--
- a_k is an 8-dimensional weight vector (one weight per landscape type). Because x_v is one-hot,
  a_k · x_v simply picks the weight of the hex's own landscape type.
- b_k is an 8-dimensional weight vector. b_k · Σ x_u adds up, for each landscape type, the number of neighbours
  of that type times its weight.
- c_k is a single number (scalar), the bias.-->

All 113 hexes share the same weights a_k, b_k, c_k (the same 16 "filters" applied at every hex). The layer therefore
has 16 neurons × 8 weights in a_k + 16 neurons × 8 weights in b_k + 16 biases c_k, i.e.
16·8 + 16·8 + 16 = 272 parameters, independent of the number of hexes. Because the neighbours are summed, the layer
does not know in which direction a neighbour lies. The first hidden layer thus has 113 × 16 neurons.

**Pooling.** For each of the 16 channels, the mean and the maximum over all hexes are taken (32 values, no weights).
This makes the model independent of the river layout: it contains no weights tied to particular hexes.

**Second hidden layer.** 8 fully connected neurons (ReLU) on top of the 32 pooled values. The network could use them
to compute one value per landscape type; whether it actually does so cannot be controlled.

**Output.** One neuron with a sigmoid, a score between 0 and 1 for "made by the smart generator"
(written P(smart) below). In total the model has 545 parameters.

The score can be read as a probability only under the conditions it was trained for: maps that come from one of the
two generators, with both equally likely. Under these conditions it is well calibrated: on 20,000 fresh maps,
for example, 10.6 % of the maps with a score between 0.05 and 0.20 were smart (mean score 0.109), 49 % of those
between 0.4 and 0.6, and 97 % of those between 0.95 and 0.99 (expected calibration error 0.002; slightly
overconfident between 0.6 and 0.8: mean score 0.71, actually 62 % smart). For maps from other sources, such as the
official maps or the deliberately broken maps below, it is only a score: it says which generator the map resembles
more, not a probability of anything.

## Generators

**Smart generator**: the [Terra Mystica Map Generator](https://github.com/ikmMaierBTUCS/Terra-Mystica-Map-Generator), installed as the package
`tm-mapgen` (see `requirements.txt`). <!--`tmgen/fast.py` is a 5× faster drop-in version (cached river access, tracked colour counts);
for the same random seed it produces exactly the same maps.-->

**Dumb generator** (`tmgen/dumb.py`) a reduced version of the smart generator that only observes that 1) two hexes of the same colour are never adjacent and
2) the remaining colours are weighted with the smart generator's factor `contingent^0.5`.
<!--It walks over the map in the same spiral order from the same start hex as the smart generator.-->

The final tests of the smart generator (`number_test`, `river_access_test`, `luck_test`) are not used for either version.

These configurations are meant to avoid distinction between dumb and smart maps by too obvious patterns.

<!--
switched off for both
generators to speed up data generation (`--tests` switches them on again). 
The colour counts are therefore not
perfectly equal between the generators: smart maps contain somewhat less blue, and colour counts alone identify the
generator in 73–76 % of the cases. To make sure that the network does not just count colours, we additionally evaluate
on a **count-matched set**: for every smart validation map we searched, among 20,000 freshly generated dumb maps,
one with exactly the same number of hexes per colour. On this set colour counts carry no information.-->

## Training

All maps use the river layout of the original Terra Mystica map. 5,000 maps of the smart and 5,000 of the dumb
generator are generated and learned (batch size 64, Adam with learning rate 0.001, binary cross-entropy, at most
600 epochs). After every epoch the loss on 1,000 + 1,000 validation maps is measured; if it has not improved for
30 epochs, training stops and the network is reset to its best epoch (**early stopping**). The validation maps are
used only for this purpose.

Then **hard example mining** is applied (`recognizer/pipeline.py`, run by `train.py`):

1. A fresh evaluation set of 5,000 + 5,000 maps is generated and the model is evaluated on it.
2. All evaluation maps that were judged wrongly (FP or FN) go into a new training set. It is filled up with correctly
   judged maps of the same evaluation set (freshly generated, never trained on), so that the wrongly judged maps make
   up 5 %. No map of the original training set is used again.
3. The existing model is saved and then trained further on the new training set (learning rate 0.0003, at most
   50 epochs, early stopping after 8).
4. A new evaluation set of 5,000 + 5,000 maps is generated, and the old and the further-trained model are both
   evaluated on it.
5. If the further-trained model makes fewer errors, the old model is discarded, the further-trained model is saved,
   and we go back to step 2 with the errors of the new model on the newest evaluation set.
   If it does not make fewer errors, further training is attempted a second time, with new data: the errors of the
   saved model on the newest evaluation set. If the further-trained model cannot beat the old model in the second
   attempt either, the old model is returned as fully trained (at most 10 rounds).

No model is ever evaluated on maps it was trained on (unless the same map is generated twice by accident). 
<!--A significance test (McNemar) is computed and logged but not
required: if the new model makes fewer errors on unseen maps, it is more likely better than worse.-->
<!--Finally, the model before mining (start model) and the model after the last accepted round (end model) are compared
on a separate fresh set of 10,000 + 10,000 maps that took part in no mining decision. This measures the overall
effect of mining without the selection effect of the rounds (a model accepted in a round partly owes its lower
error count on that round's set to chance). The same set was then used to choose the final model among the six
trained models (2 channel widths × 3 seeds).-->

A held-out test set of 2,000 + 2,000 maps is used only once, for the final model.

## Results

The trained model distinguishes maps of the smart generator from maps of the dumb generator with an accuracy of
**98.1 %** (held-out test set of 2,000 + 2,000 maps, 95 % confidence interval 97.7–98.6 %).
The FP rate (dumb map judged smart) is **1.9 %**, the FN rate (smart map judged dumb) **1.85 %**.

The model was trained on maps of version 4 of the map generator. Version 4.1 fixes the luck bookkeeping of the first
hex after the start field, which changes most maps slightly. The model recognises them just as well: of 2,000 smart
maps generated with the test seeds, 98.6 % (version 4.1) and 98.3 % (version 4) are judged smart.

<!--| | 16 channels | 8 channels |
|---|---|---|
| Validation accuracy before mining (3 seeds) | 0.980 ± 0.003 | 0.963 ± 0.002 |
| Errors on 20,000 fresh maps before → after mining (3 seeds) | 422 → 366, 436 → 392, 429 → 394 | 750 → 718, 744 → 645, 757 → 725 |
| Test accuracy of the final model (16 channels, seed 0, after mining) | **0.981** | – |

16 channels per hex are clearly better than 8 on the original layout. The final model was chosen as the one with
the fewest errors on the 20,000-map comparison set; the test set was evaluated only once, for this model.-->

**Official maps.** 
The score P(smart) is calculated for all official maps.
The model's accuracy on 1,000 + 1,000 generated maps of the same layout
is given for context. Only the original layout has been used for training.

| Map | Score P(smart) | Accuracy on generated maps of this layout |
|---|---|---|
| Original | **0.999** | 0.977 |
| Revised | **0.990** | 0.977 |
| Fjords | **1.000** | 0.980 |
| Fire & Ice | **0.661** | 0.980 |
| Loon Lakes | **1.000** | 0.984 |

All official maps are assigned to the smart generator. Fire & Ice is the only one the model is unsure about:
it has more neighbours one step apart on the colour wheel (43) than typical smart maps of its layout (41.3; dumb: 43.5).
<!--The model also works on the Fire & Ice and Loon Lakes layouts, whose first row is short (112 hexes), which it has never
seen during training.

**Transfer to other river layouts.** A model trained only on the Fjords layout recognised maps of the original
layout, which it had never seen, with 98.2 % accuracy (98.5 % on Fjords). This is a benefit of pooling instead of
position-specific weights.

**Mining.** Mining reduced the errors of all six models; for the 16-channel models by 8–13 % (significant in each
case, McNemar p ≤ 0.009), for the 8-channel models by 4–13 %.-->

**The model can be tricked.** If two hexes of a balanced map are swapped so that a pair of adjacent hexes of the same
colour appears, the model still calls the map smart (8 of 8 maps, mean P(smart) = 0.94). After all, the model has never
seen adjacent hexes of the same colour in its training data; it only ranks maps between smart and dumb.
Same-colour pairs push its judgement towards dumb, but with 5 pairs it still judges 4 out of 8 maps smart. 
Conversely, making a dumb map look smart by having less blue adjacent to rivers - a pattern observed for smart maps - did not fool the final model (0 of 8).
It is therefore recommended to first check whether a map 1) has no adjacent hexes of the same colour and 2) passes
`number_test` and `river_access_test`. If these conditions are met and the trained model classifies the map as smart,
this may indicate a balanced map. `map_is_smart.py` prints a warning if one of these checks fails. Note that
`river_access_test` is strict: of the official maps, only the revised map passes it.

<!--### How we arrived at the architecture

All numbers are validation accuracy on the Fjords layout (mean over 3 seeds unless stated otherwise).

| Experiment | Result |
|---|---|
| Baselines | colour counts only 0.73; hand-made features + gradient boosting 0.835 |
| Model type | MLP 0.73 · CNN 0.963 · hexagonal CNN 0.963 · GCN (3 graph layers × 32) 0.986 |
| Colour encoding (hexagonal CNN) | one-hot 0.963 · circular (sin, cos) 0.928 · index k/6 0.810 |
| Activation (1 graph layer × 8) | ReLU 0.978 · LeakyReLU 0.978 · quadratic 0.971 |
| Channels per hex (1 graph layer) | 7: 0.970 · 8: 0.978 · 16: 0.980 · 32: 0.971 (overfitting) |
| Dense layers after pooling | 8: 0.978 · 113: 0.968 · 113 → 8: 0.971 |
| Count-matched set (GCN 3 × 32) | 0.983, i.e. the network judges the arrangement, not the colour counts |
| Error analysis | circular and index encodings misjudge blue systematically; one-hot shows no systematic error |

### Limitations

- Trained only on original river layout.
- Validation accuracies are slightly optimistic because the validation set is also used for early stopping;
  the held-out test set gives the unbiased number.
- The error analysis is based on one seed per model.-->

## Repository

```
map_is_smart.py   check a map with the trained model (numpy only)
train.py          train your own model from two generator files -> model.py
generators/       smart_generator.py, dumb_generator.py (generator files for train.py)
tmgen/            dumb.py (dumb generator), fast.py (fast drop-in versions of both generators)
recognizer/       model_template.py (the code of map_is_smart.py / model.py), export.py (writes model.py),
                  pipeline.py (training and hard example mining), genload.py (loads generator files), gcn.py (GCN)
```

## Usage

Requires Python 3.8 or newer.

### Check a map

Needs only numpy.

```bash
python map_is_smart.py map.txt              # prints a score between 0 and 1, e.g. 0.999
```

`map.txt` contains one map row per line, hexes separated by commas, as written by the map generator and as uploaded
to Board Game Arena (R red, Y yellow, U brown, K black, B blue, G green, S gray, I river). The snellman format (rows separated by `;`, a leading `;` for a short first row) works as well. Any map size is accepted.

### Train your own model with your own map generator

Needs numpy, torch and the map generator package tm-mapgen (`pip install -r requirements.txt`).

A Python random map generator smart_generator.py is needed. A dumb version for comparison is also needed but you may just use generators/dumb_generator.py.

```bash
python train.py generators/smart_generator.py generators/dumb_generator.py --out model.py
```

train.py may take a few hours. It returns a script model.py that can be used the same way as map_is_smart.py:

```bash
python model.py map.txt                     # used exactly like map_is_smart.py
```
