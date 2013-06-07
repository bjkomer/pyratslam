import pstats
import sys

def main():
  if len( sys.argv ) < 2:
    print("No file specified. Exiting...")
    return 1
  stat_file = sys.argv[1] # The first argument is the script itself, need to grab the second
  sort = 'cumulative'
  sort_options = [ 'calls', 'cumulative', 'cumtime', 'file', 'filename', 'module', 'ncalls', 'pcalls',
                   'line', 'name', 'nfl', 'stdname', 'time', 'tottime' ]
  if len( sys.argv ) >= 3:
    sort = sys.argv[2]
    if sort not in sort_options:
      sort = 'cumulative'
  stats = pstats.Stats(stat_file)

  stats.sort_stats(sort)

  stats.print_stats(.10)

if __name__ == "__main__":
  main()
