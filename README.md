# bd_assignment1

## Article

https://hackmd.io/@tootiredone/search-engine

## Commands

```
# running Indexer
hadoop jar Indexer.jar Indexer /EnWikiSmall Indexer_new_small

# running Query
hadoop jar Query.jar Query "theory of algorithms" 10 /EnWikiSmall Indexer_new_small Query_new_small
```

## Structure

### input
part of docs copied for testing 

### yes okapi

Implementation of the search engine using okapi

### no okapi

Implementation of the search engine with simple formulas

## Testing

Both implementation were tested on `/EnWikiSmall` and `/EnWikiMedium`

