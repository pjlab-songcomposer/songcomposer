# SongComposer  Finetuning

We offer the official scripts for easy finetuning of the pretrained songcomposer model on downstream tasks. Our finetune scripts use DeepSpeed and FSDP by default.

### Environment Setup

**Step 1.** Create a conda environment and activate it.

```bash
conda create -n songcomposer python=3.9 -y
conda activate songcomposer
```

**Step 2.** Install PyTorch (We use PyTorch 2.1.2 / CUDA 12.0)
```bash
pip3 install torch torchvision torchaudio
```

**Step 3.** Install require packages

```bash
pip install transformers==4.31.0 timm==0.6.13 sentencepiece==0.1.99 gradio==4.13.0 markdown2==2.4.10 xlsxwriter==3.1.2 einops pretty_midi
```

**Step 4.** Install deepspeed for fine-tuning

```bash
pip install deepspeed
```

### Data preparation

To prepare your finetuning data, you should (1) formulate each sample as a dictionary consisting of a question, a answer, and (2) save data samples in JSON files.

**Format Notation**

\<bop\> stands for the **b**eginning **o**f the **p**air. \<eop\> stands for the **e**nd **o**f the **p**air.

\<bom\> stands for the **b**eginning **o**f the **m**elody. \<eom\> stands for the **e**nd **o**f the **m**elody.

\<bol\> stands for the **b**eginning **o**f the **l**yric. \<eol\> stands for the **e**nd **o**f the **l**yric.

The conversation format would be:
```
[UNUSED_TOKEN_146]user\n{Question}[UNUSED_TOKEN_145]\n

[UNUSED_TOKEN_146]assistant\n{Answer}[UNUSED_TOKEN_145]\n
```
For the detailed construction of melody, text, melody-text pair please refer to our paper.

<details>
  <summary>
    <b>finetune example list (example_data.json) with 4 samples.</b>
  </summary>

```
  [
    {
      'question': 'Given the following lyrics, create a suitable melody. <bol> Total 6 lines. The first line:对|不|起|我|却|没|戳|紧|你\n The second line:你|不|知|道|我|为|什|么\n The third line:离|开|你|我|坚|持|不|能|说|放|任|你|哭|泣\n The fourth line:你|的|泪|滴|像|倾|盆|大|雨\n The fifth line:碎|了|满|地\n The sixth line:在|心|里|清|晰\n<eol>',
      'answer': '<bop> Total 6 lines. The first line:对,<A3>,<143>,<81>|不,<A3> <C4>,<143><119>,<79>|起,<B3>,<171>,<185>|我,<F#3>,<121>,<81>|却,<C4> <D4>,<153><130>,<81>|没,<B3>,<141>,<84>|戳,<D#4> <E4>,<145><127>,<79>|紧,<B3>,<162>,<81>|你,<B3> <C4>,<178><202>,<156>\n The second line:你,<D4>,<159>,<81>|不,<E4> <F4>,<128><117>,<79>|知,<G4>,<140>,<81>|道,<A4>,<139>,<84>|我,<C#4>,<139>,<81>|为,<A4> <A#4>,<126><123>,<81>|什,<A4>,<172>,<79>|么,<A4>,<130>,<110>\n The third line:离,<F4> <F#4>,<139><122>,<81>|开,<F4> <E4>,<175><130>,<79>|你,<E4>,<137>,<142>|我,<A3>,<127>,<81>|坚,<F4>,<152>,<81>|持,<G4>,<139>,<81>|不,<G#4>,<132>,<79>|能,<G#3>,<125>,<81>|说,<G#4> <A#4>,<158><119>,<84>|放,<G#4>,<159>,<81>|任,<F4> <G4> <F4>,<125><115><130>,<81>|你,<F4>,<127>,<81>|哭,<D#4> <E4>,<166><136>,<79>|泣,<D4>,<163>,<117>\n The fourth line:你,<A3>,<144>,<79>|的,<C#4>,<149>,<79>|泪,<F4>,<135>,<81>|滴,<G4>,<130>,<84>|像,<D#4>,<142>,<81>|倾,<F#4> <G4>,<139><142>,<81>|盆,<F#4>,<151>,<81>|大,<E4> <D4>,<145><130>,<81>|雨,<D4>,<146>,<122>\n The fifth line:碎,<D4> <C#4>,<189><108>,<81>|了,<C#4>,<114>,<79>|满,<C#4> <D4>,<155><142>,<81>|地,<E4>,<151>,<122>\n The sixth line:在,<F4> <F#4>,<137><122>,<81>|心,<F#4>,<168>,<79>|里,<F#4>,<149>,<84>|清,<E4> <F#4>,<168><121>,<81>|晰,<F#4>,<190>,<157>\n<eop>'
    },
    {
      'question': 'Compose a song about global unity and building a hopeful future together.',
      'answer': 'The song is as follows. <bop> Total 6 lines. The first line:这,<C4>,<110>,<92>|地,<A#3> <G#4>,<233><92>,<130>|球,<G4>,<120>,<273>|会,<G4>,<134>,<79>|合,<G4>,<134>,<123>|唱,<C4>,<205>,<166>\n The second line:齐,<C4>,<92>,<79>|心,<C4>,<92>,<79>|唤,<C4>,<92>,<79>|醒,<C4>,<92>,<100>|心,<C4>,<200>,<79>|中,<A#3>,<141>,<104>|那,<G3>,<141>,<104>|火,<A#3>,<194>,<100>|太,<G3>,<120>,<194>|阳,<C3>,<108>,<79>\n The third line:携,<D#3>,<93>,<88>|手,<C4>,<200>,<172>|为,<D4>,<104>,<120>|建,<C#4>,<141>,<104>|造,<C4>,<137>,<79>|未,<A#3> <G#3>,<157><154>,<88>|来,<A#3> <C4>,<207><144>,<79>|齐,<C#4>,<189>,<79>|奉,<C#4>,<189>,<79>|献,<C#4>,<189>,<79>\n The fourth line:始,<C4>,<141>,<79>|终,<A#3>,<151>,<96>|有,<G3>,<120>,<79>|一,<G#3>,<108>,<79>|天,<C4>,<112>,<79>\n The fifth line:这,<C4>,<112>,<96>|地,<A#3>,<237>,<189>|球,<G#3>,<123>,<104>|会,<G3>,<137>,<96>|合,<F3>,<137>,<96>|唱,<G3> <G#3> <G3>,<137><127><175>,<130>\n The sixth line:齐,<G#3>,<334>,<194>|心,<D#3> <C#4>,<147><202>,<160>|唤,<C#4>,<141>,<88>|醒,<C#4>,<134>,<100>|心,<C#4>,<147>,<79>|中,<B3>,<144>,<92>|那,<G#3>,<166>,<79>|火,<A#3>,<197>,<79>|太,<G#3>,<166>,<79>|阳,<C#3>,<266>,<79>\n<eop>'
    },
  ]
```

</details>


After data preparation, you can use the provided bash scripts (`finetune.sh`) to finetune the model. Remember to specify the pre-train model path ($MODEL) and the txt data file path ($DATA) in the bash script.

### Full-parameter finetuning

Full-parameter parameter finetuning requires updating all parameters of LLM in the whole training process. To launch your training, run the following script:

```
sh finetune.sh
```
