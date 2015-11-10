# Worthy Opponent

`worthy-opponent` is a program that plays two-player [abstract strategy games](https://en.wikipedia.org/wiki/Abstract_strategy_game). For now it just plays against itself.

## Games

Currently three games are implemented: [Tic-tac-toe](https://en.wikipedia.org/wiki/Tic-tac-toe), [Nine Men's Morris](https://en.wikipedia.org/wiki/Nine_Men%27s_Morris), [Go](https://en.wikipedia.org/wiki/Go_(game)), and [Connect Four](https://en.wikipedia.org/wiki/Connect_Four). Soon it will play [every other game ever](https://en.wikipedia.org/wiki/General_game_playing).

## How it works

`worthy-opponent` uses [Monte Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) to pick its moves. It plays random games, keeping track of how likely each move is to result in a win, then selects the best move when asked to play.

## How to run it

    cargo run --release --bin two-player -- playername [server:port]

The default value for `server:port` is `127.0.0.1:11873`.

The server in the `worthy-server` folder is currently out of date. Use the server from [@NickLarsen's repository](https://github.com/NickLarsen/game-frame) until I update it.
