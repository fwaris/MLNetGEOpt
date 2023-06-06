module GenomeSize
open System
open MLNetGEOpt

let run() =
    let rng = Random()
    let g = Game.grammar
    let gbase = Grammar.estimateGenomeSize g
    let genome = Array.zeroCreate 11
    let simSzs = 
        [for j in 0 .. 10000 do
            for i in 0 .. genome.Length-1 do
                genome.[i] <- rng.Next(256)
            let ts,i = Grammar.translate g genome
            // "%d" i
            i]
    printfn $"gsim max {List.max simSzs}"
    printfn "%A" gbase
    ()