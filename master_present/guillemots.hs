#!/bin/env runhaskell

import Text.Pandoc.JSON

main :: IO ()
main = toJSONFilter guillemots
  where guillemots (Just (Format fmt)) (Quoted DoubleQuote str) | fmt == "latex" || fmt == "beamer" = 
            [RawInline latex "{<<}"] ++ str ++ [RawInline latex "{>>}"] where latex = Format "latex"
        guillemots (Just html@(Format "html")) (Quoted DoubleQuote str) = 
            [RawInline html "&laquo;"] ++ str ++ [RawInline html "&raquo;"]
        guillemots _ x = [x]

