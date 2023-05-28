﻿namespace MLNetGEOpt
open CA
open Microsoft.ML.AutoML
open Microsoft.ML

module Optimize =
    (*
        evolved the genome using CA rules
        convert CA individual to sweepable pipeline
        evaluate sweepable pipeline
        use cache to speed up processing
        feedback fitness to CA for next gen evolution

        What is the fitness function here?
        We leverage AutoMLExperiment. We need:
        - dataset
        - sweeplable pipeline (see above)
        - trialRunner
        - metric 
          - binary classification
          - multi-class classfication
          - regression         
    *)
    let run kind (expFac:SweepablePipeline -> AutoMLExperiment) (g:Grammar)=
        let worstVal = match kind with Minimize -> 9e10 | _ -> -9e10
        let genomSize = Grammar.esimateGenomeSize g
        let genomSize = if genomSize < 4 then genomSize else genomSize / 2
            
        let parms = [1 .. genomSize] |> List.map(fun x -> I(0,255,128)) |> List.toArray

        let mutable fmap = Map.empty
        let bestRslt = ref Unchecked.defaultof<_>

        let setResult (t:TrialResult) =
            if bestRslt.Value = Unchecked.defaultof<_> then
                bestRslt.Value <- t
            else
                let vPrev = bestRslt.Value.Metric
                let vNnew = t.Metric
                match kind with
                | Maximize -> if vNnew >= vPrev then bestRslt.Value <- t
                | Minimize -> if vNnew <= vPrev then bestRslt.Value <- t

        let rsltAgnt = MailboxProcessor.Start(fun inbox -> 
            async {
                while true do
                    let! msg = inbox.Receive()
                    setResult msg
            })

        let fitness (pvals:float[]) = 
            let genome = pvals |> Array.map int
            fmap
            |> Map.tryFind genome
            |> Option.defaultWith(fun _ -> 
                let terminals,_ = Grammar.translate g genome
                let pipeline = Grammar.toPipeline terminals
                let exp = expFac pipeline                
                try 
                    let rslt = exp.Run()
                    fmap <- fmap |> Map.add genome rslt.Metric
                    rsltAgnt.Post rslt
                    rslt.Metric
                with ex ->
                    printfn $"Ex {ex.Message}"
                    printfn $"%A{Grammar.printPipeline (MLContext()) terminals}"
                    worstVal                    
                )
             
        let mutable step = CALib.API.initCA(parms, fitness, kind,36)
        for i in 0 .. 15000 do 
            step <- CALib.API.Step step
        
        let gbest = step.Best.[0].MParms |> Array.map int
        let gbestTerms,_ = Grammar.translate g gbest 
        gbestTerms, step.Best.[0].MFitness,bestRslt.Value
