# <img src="img/logo.png" style="vertical-align: -10px;" :height="40px" width="40px"> SongComposer
This repository is the official implementation of SongComposer.

<!-- **[SongComposer: A Large Language Model for Lyric and Melody Composition in Song Generation](https://arxiv.org/abs/2402.17645)**
</br> -->
<p align="center" style="font-size: 1.5em; margin-top: -1em"> <a href="https://arxiv.org/abs/2402.17645"><b>SongComposer: A Large Language Model for Lyric and Melody Composition in Song Generation</b></a></p>
<p align="center" style="font-size: 1.1em; margin-top: -1em">
<a href="https://mark12ding.github.io/">Shuangrui Ding<sup>*1</sup></a>,  
<a href="https://scholar.google.com/citations?user=iELd-Q0AAAAJ">Zihan Liu<sup>*2,3</sup></a>,  
<a href="https://scholar.google.com/citations?user=FscToE0AAAAJ">Xiaoyi Dong<sup>3</sup></a>,  
<a href="https://panzhang0212.github.io/">Pan Zhang<sup>3</sup></a>,  
<a href="https://shvdiwnkozbw.github.io/">Rui Qian<sup>1</sup></a>,  
<a href="https://conghui.github.io/">Conghui He<sup>3</sup></a>,  
<a href="http://dahua.site/">Dahua Lin<sup>3</sup></a>,  
<a href="https://myownskyw7.github.io/">Jiaqi Wang<sup>&dagger;3</sup></a> 
</p>
<p align="center" style="font-size: 1em; margin-top: -1em"><sup>1</sup>The Chinese University of Hong Kong, <sup>2</sup>Beihang University, <sup>3</sup>Shanghai AI Laboratory</p>
<p align="center" style="font-size: 1em; margin-top: -1em"> <sup>*</sup>  Equal Contribution. <sup>&dagger;</sup>Corresponding authors. </p>

<p align="center" style="font-size: em; margin-top: 0.5em">
<a href="https://arxiv.org/abs/2402.17645"><img src="https://img.shields.io/badge/arXiv-<color>"></a>
<a href="https://github.com/pjlab-songcomposer/songcomposer"><img src="https://img.shields.io/badge/Code-red"></a>
<a href="https://pjlab-songcomposer.github.io"><img src="https://img.shields.io/badge/Demo-yellow"></a>
</p>

<img align="center" src="img/framework.png" style="  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 100%;" />

## 📜 News

🚀 [2023/2/28] The [paper](https://arxiv.org/abs/2402.17645) and [demo page](https://pjlab-songcomposer.github.io) are released!

## 💡 Highlights
- 🔥SongComposer composes melodies and lyrics with symbolic song representations, with the benefit of
**better token efficiency**, **precise representation**, **flexible format**, and **human-readable output**.
- 🔥  SongCompose-PT, a comprehensive pretraining dataset that includes lyrics, melodies, and
paired lyrics and melodies in either Chinese or English, will be released.
- 🔥 SongComposer outperforms advanced LLMs like GPT-4 in tasks such as lyric-to-melody generation, melody-to-lyric generation, song continuation, and text-to-song creation.

## 👨‍💻 Todo
- [ ] Training code for SongComposer
- [ ] Evaluation code for SongComposer
- [ ] Checkpoints of SongComposer
- [x] Demo of SongComposer


## 🛠️ Usage
Updating in progress...

##   ⭐ Demos
For more details, refer to the [Demo Page](https://pjlab-songcomposer.github.io).
### Lyric-to-Melody
   **Given Lyrics:** 轻松踏上我的路 有人向我打招呼 欢迎陪我看日出 没有包袱只有礼物 未来由我来建筑  <br>
   **English translation:** Stepping onto my path with ease, someone greets me. Welcome to accompany me to watch the sunrise—no burdens, only gifts. The future is to be built by me.<br>
   **Pinyin:** Qīngsōng tà shàng wǒ de lù, yǒu rén xiàng wǒ dǎzhāohū. Huānyíng péi wǒ kàn rìchū, méiyǒu bāofù zhǐyǒu lǐwù. Wèilái yóu wǒ lái jiànzhù.
    <table style='width: 100%;'>
        <thead>
        <tr>
            <th></th>
            <th>Ground Truth</th>
            <th>GPT-3.5</th>
            <th>GPT-4</th>
            <th>SongComposer(Ours)</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <th scope="row">Wav</th>
            <td><audio controls="" ><source src="showcase/l2m/zh/1/gt.wav" type="audio/wav"></audio></td>
            <td><audio controls="" ><source src="showcase/l2m/zh/1/gpt3.5.wav" type="audio/wav"></audio></td>
            <td><audio controls="" ><source src="showcase/l2m/zh/1/gpt4.wav" type="audio/wav"></audio></td>
            <td><audio controls="" ><source src="showcase/l2m/zh/1/ours.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody>
    </table>

### Melody-to-Lyric
   **Given melody:** <audio controls="" ><source src="showcase/m2l/en/2/melody.wav" type="audio/wav"></audio>
    <table style='width: 100%;'>
        <thead>
        <tr>
            <th></th>
            <th>Ground Truth</th>
            <th>GPT-3.5</th>
            <th>GPT-4</th>
            <th>SongComposer(Ours)</th>
        </tr>
        </thead>
        <tbody>
           <tr>
            <th scope="row">Wav</th>
            <td><audio controls="" ><source src="showcase/m2l/en/2/gt.wav" type="audio/wav"></audio></td>
            <td><audio controls="" ><source src="showcase/m2l/en/2/gpt3.5.wav" type="audio/wav"></audio></td>
            <td><audio controls="" ><source src="showcase/m2l/en/2/gpt4.wav" type="audio/wav"></audio></td>
            <td><audio controls="" ><source src="showcase/m2l/en/2/ours.wav" type="audio/wav"></audio></td>
        </tr>
        <tr>
            <th scope="row">Lyrics</th>
            <td><p>Getting stronger, coming closer, that time for me to face it though this might just be the ending of the life I held so dear.</p>
            </td>
            <td><p>Oh, how I wish you were here with me tonight. So I could show you the way I feel inside when I see you smiling.</p>
            </td>
            <td><p>Rising colors painting skies up above, high-flying birds soar wide. Freedom calls to us now as we reach out towards new heights.</p>
            </td>
            <td><p>Where the wildflowers grow and the sun comes over, and a bird sits in the trees, it is a good day.</p>
            </td>
        </tr>
    </tbody>
    </table>

### Song Continuation
  **Given song prompt:** <audio controls="" ><source src="showcase/sc/zh/3/prompt.wav" type="audio/wav"></audio>
    <table style='width: 100%;'>
        <thead>
        <tr>
            <th></th>
            <th>Ground Truth</th>
            <th>GPT-3.5</th>
            <th>GPT-4</th>
            <th>SongComposer(Ours)</th>
        </tr>
        </thead>
        <tbody>
         <tr>
            <th scope="row">Wav</th>
            <td><audio controls="" ><source src="showcase/sc/zh/3/gt.wav" type="audio/wav"></audio>
            </td>
            <td><audio controls="" ><source src="showcase/sc/zh/3/gpt3.5.wav" type="audio/wav"></audio>
            </td>
            <td><audio controls="" ><source src="showcase/sc/zh/3/gpt4.wav" type="audio/wav"></audio>
            </td>
            <td><audio controls="" ><source src="showcase/sc/zh/3/ours.wav" type="audio/wav"></audio>
            </td>
        </tr>
        <tr>
            <th scope="row">Lyrics</th>
            <td><p>闻鹧鸪声声催人别回头 怎么记载往事的对白 只留一纸独白 怎么离开</p>
            </td>
            <td><p>闻鹧鸪声声催人别回头 月光照亮夜空明星点 风吹过草原 野青翠绿 花香四溢 五色斑斓 七此时刻最美忆留</p> 
            </td>
            <td><p>闻鹧鸪声声催人别回头 远处炊烟起 浮想那片青楼瓦 懒挥扇影动斜阳内 忽闻歌声传来意味</p> 
            </td>
            <td><p>闻鹧鸪声声催人别回头 伤春客醉后回忆惆怅 却不如回首 前世外一场梦</p>
            </td>
        </tr>
        <tr>
            <th scope="row">English Translation</th>
            <td><p>Hearing the sound of magpies, urging people not to look back. How to record the dialogue of the past, leaving only a monologue on paper. How to leave.</p>
            </td>
            <td><p>Hearing the sound of the chuckling partridge urges one on, don't look back. Moonlight illuminates the night sky, with stars twinkling. The wind blows across the grassland, wild and green. The fragrance of flowers permeates the air, displaying a colorful and vibrant scene. This moment, the seventh, is the most beautiful, etched in memory.</p> 
            </td>
            <td><p>Hearing the melodious sound of the chuckling partridge urges one on, don't look back. Distant cooking smoke rises, thoughts drifting to that tiled blue building. Leisurely waving a fan, shadows moving in the slanting sun, Suddenly, the sound of singing comes with meaning.</p> 
            </td>
            <td><p>Hearing the sound of the chuckling partridge urges one on, don't look back. The wounded spring, memories after a drunken return, are melancholic. However, it is not as good as looking back, as if the past life was just an ephemeral dream.</p>
            </td>
        </tr>
        <tr>
            <th scope="row">Pinyin</th>
            <td><p>Wén zhègū shēng shēng cuī rén bié huítóu. Zěnme jìzǎi wǎngshì de duìbái, zhǐ liú yī zhǐ dúbái. Zěnme líkāi.</p>
            </td>
            <td><p>Wén zhè gū shēng, shēng cuī rén bié huítóu. Yuèguāng zhào liàng yè kōng, míngxīng diǎn. Fēng chuīguò cǎoyuán, yě qīngcuì lǜ. Huā xiāng sì yì, wǔsè bānlán. Qī cǐ shíkè zuì měi, yì liú.</p>
            </td>
            <td><p>Wén zhè gū shēng shēng cuī rén bié huítóu, Yuǎnchù chuīyān qǐ, Fú xiǎng nà piàn qīnglóu wǎ, Lǎn huī shàn yǐng dòng xiéyáng nèi, Hū wén gēshēng chuánlái yì.</p>
            </td>
            <td><p>Wén zhè gū shēng, shēng cuī rén bié huítóu. Shāng chūn kè zuì hòu huíyì chóuchàng. Què bùrú huíshǒu, qiánshì wài yī chǎng mèng.</p>
            </td>
        </tr>
    </tbody>
    </table>
### Text-to-Song
   **Given Text:** Create a song on brave and sacrificing with a rapid pace.
    <table style='width: 100%;'>
        <thead>
        <tr>
            <th></th>
            <th>GPT-3.5</th>
            <th>GPT-4</th>
            <th>SongComposer(Ours)</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <th scope="row">Wav</th>
            <td><audio controls="" ><source src="showcase/t2s/en/2/gpt3.5.wav" type="audio/wav"></audio></td>
            <td><audio controls="" ><source src="showcase/t2s/en/2/gpt4.wav" type="audio/wav"></audio></td>
            <td><audio controls="" ><source src="showcase/t2s/en/2/ours.wav" type="audio/wav"></audio></td>
        </tr>
        <tr>
            <th scope="row">Lyrics</th>
            <td><p>Brave hearts unite in this fight tonight. We stand together under the bright moonlight.</p>
            </td>
            <td><p>Brave is what you are, facing adversity without a scar. Sacrifice your joy and pain.</p>
            </td>
            <td><p>Brave enough to let you go, faithful enough to be the hero, you are the reason why I cried.</p>
            </td>
        </tr>
    </tbody>
    </table>

## ❤️ Acknowledgments


## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝
```bibtex
@misc{ding2024songcomposer,
      title={SongComposer: A Large Language Model for Lyric and Melody Composition in Song Generation}, 
      author={Shuangrui Ding and Zihan Liu and Xiaoyi Dong and Pan Zhang and Rui Qian and Conghui He and Dahua Lin and Jiaqi Wang},
      year={2024},
      eprint={2402.17645},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## License

