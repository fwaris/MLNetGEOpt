module GenomeSize
open System
open MLNetGEOpt

let run() =
    let rng = Random()
    let g = Game.grammar
    let genome = Array.zeroCreate 11
    for j in 0 .. 20 do
        for i in 0 .. genome.Length-1 do
            genome.[i] <- rng.Next(256)
        let ts,i = Grammar.translate g genome
        printfn "%d" i
    ()