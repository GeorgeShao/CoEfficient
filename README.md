# CoEfficient
Climate change and global warming are counted amongst the foremost threats to humanity. While we struggle to find a way to reverse climate change, we must do our best to help slow it as best as we can. To that effect, we thought of focusing on the tourism transportation and travel industry.

The problem is that every year, vehicles waste gas, time, and money by driving extra distances when there is often a shorter route available. Every year, 8 trillion tons of carbon dioxide are emitted into the atmosphere from transportation sources. If we could reduce our trip lengths by as little as 0.5% when there are shorter routes available, we could reduce our carbon dioxide emissions by 40 million tons!

Our project solves that problem.

## What it does
CoEfficient is an application that finds the most efficient route between multiple points on a map. This is often very useful for a variety of clients including:
- coach bus operating businesses who are looking to give their vacationing passengers a great time visiting many landmarks but also want to take the most efficient route possible
- school bus operating businesses who are seeking to pick-up/drop-off children at multiple points in the most efficient way possible
- travelling individuals who want to visit multiple landmarks and spend more time at each landmark instead of wasting time driving in between landmarks
- people running errands who need to visit multiple locations as fast as possible

## How we built it
We built our app in Python. We used the Google Maps API to compute the distances in between nodes on the map. We used numpy, networkx, and matplotlib to efficiently compute and create a representative graph of the map and its nodes and edges. Finally we used Qiskit, a quantum information science kit, along with IBM-Q, a cloud-accessible quantum computer to compute the most efficient route to the problem. This would allow for exponentially more efficient mass scaling in the future.

## Challenges we ran into
Not every member of the had experience dealing with app development and even the python language. Half our team was experienced with 7+ hackathons and for the other half of the team, this was their first hackathon - leading to an interesting team dynamic. The team worked together to overcome any major hurdles to keep the work going. While we did spend some time debugging various errors the largest hurdle by far was working with Qiskit, and getting the quantum computer code to compile and run on the IBM-Q quantum cloud computer.

## Accomplishments we are proud of
Our team is very proud of learning about and using IBM-Q's quantum cloud computer along with our efficient node relative distance and latitude and longitude matching algorithms.

## What's next for CoEfficient
The current version of the app can be refined by taking traffic and speed limits into account when calculating the most efficient route. In the future, this project could be modified to suit air and water travel for airplanes and shipping companies.