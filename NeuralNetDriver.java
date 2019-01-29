package neuralnet;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.regex.Pattern;
import java.util.concurrent.TimeUnit;
import java.io.*;

public class NeuralNetDriver
{
	public static void main(String[] args) throws IOException
	{
		String start_file = args[0];
		
		ArrayList<Node> starting_nodes = new ArrayList<Node>();
		
		ArrayList<Double> measured_outputs = new ArrayList<Double>();
		
		ArrayList<Network> networks = new ArrayList<Network>();
		
		BufferedReader start_file_lines = new BufferedReader(new FileReader(start_file));
		
		String node_file = start_file_lines.readLine();
		
		int hiden_layers = Integer.parseInt(start_file_lines.readLine());
		
		int hiden_nodes_in_layer = Integer.parseInt(start_file_lines.readLine());
		
		float learning_num = Float.parseFloat(start_file_lines.readLine());
		
		float allowed_time = Float.parseFloat(start_file_lines.readLine());
		
		int allowed_time_in_sec = Math.round(60 * allowed_time);
		
		//BufferedReader node_file_lines = new BufferedReader(new FileReader(node_file));
		
		Scanner node_file_lines = new Scanner(new File(node_file));
		String[] lineVector;
		String line;
		
		double max = node_file_lines.nextDouble();
		
		
		line = node_file_lines.next();
		
		lineVector = line.split(",");
		
		int num_inputs = Integer.parseInt(lineVector[0]);
		
		int num_outputs = Integer.parseInt(lineVector[1]);
		
		int total_puts = num_inputs + num_outputs;
		
		while(node_file_lines.hasNext())
		{
			line = node_file_lines.next();
			lineVector = line.split(",");
			for(int i = 0; i < num_inputs; i++)
			{
				starting_nodes.add( new Node(Double.parseDouble(lineVector[i]),false,false));
			}
			
			for(int i = num_inputs; i < total_puts; i++)
			{
				measured_outputs.add(Double.parseDouble(lineVector[i]));
			}
			
			/*input = Integer.parseInt(lineVector[0]);
			measured_output = Integer.parseInt(lineVector[1]);
			
			starting_nodes.add( new Node(input,measured_output,false,false));
			*/
		}
		
		
		if(num_inputs == 1)
		{
			for(Node starting_node : starting_nodes)
			{
				Network network = new Network();
				network.addStaringNode(starting_node);
				networks.add(network);
			}
		}
		else
		{
			int num_networks = starting_nodes.size()/num_inputs;
			
			for(int i = 0; i < num_networks; i++)
			{
				Network network = new Network();
				for(int x = 0; x < num_inputs; x++)
				{
					Node starting_node = starting_nodes.remove(0);
					network.addStaringNode(starting_node);
				}
				networks.add(network);
			}
		}
		
		
		int x = 0;
		for (Network network : networks)
		{
			int temp = x;
			for(int i = 0; i < num_outputs; i++)
			{
				network.addMeasuredOutputs(measured_outputs.get(temp));
				temp++;
			}
			x = x + num_outputs;
		}
		
		//Adding the right amount of hidden nodes
		int num_hidden_nodes = hiden_layers * hiden_nodes_in_layer;
		for (Network network : networks)
		{
			for(int i = 0; i < num_hidden_nodes; i++)
			{
				network.addConnectedNode( new Node(0,true,false));
			}
		}
		
		//Adding the right amount of outputs
		for (Network network : networks)
		{
			for(int i = 0; i < num_outputs; i++)
			{
				network.addConnectedNode( new Node(0,false,true));
			}
		}
		
		//Adding weights to connected nodes
		for (Network network : networks)
		{
			for(int i = 0; i < hiden_nodes_in_layer; i++) //connecting nodes that connect to input
			{
				network.randomizeConnectedNodeWeights(i,num_inputs+1); //account for bias weight
			}
			for(int i = hiden_nodes_in_layer; i < num_hidden_nodes; i++) //connecting nodes that connect to each other
			{
				network.randomizeConnectedNodeWeights(i,hiden_nodes_in_layer+1);
			}
		}
		
		//Adding weights to outputs
		int total_noninput_nodes = num_hidden_nodes + num_outputs;
		int output_start = total_noninput_nodes - num_outputs;
		for (Network network : networks)
		{
			for(int i = output_start; i < total_noninput_nodes; i++)
			{
				network.randomizeConnectedNodeWeights(i,hiden_nodes_in_layer+1);
			}
		}
		
		boolean test = true;
		int i = 0;
		for (long stop=System.nanoTime()+TimeUnit.SECONDS.toNanos(allowed_time_in_sec);stop>System.nanoTime();)// loop 
		{
			i++;
			for(Network network: networks)
			{
				network.setStartingNodesInputBack();
			}
			
			for(Network network: networks)//forward
			{
				network.giveConnectingNodesInput(hiden_nodes_in_layer, num_inputs);
			}
			
			for(Network network: networks)
			{
				network.giveOutputNodesJValue(num_outputs,max);
			}
			for(Network network: networks)
			{
				network.giveHiddenLayerNodesJValue(hiden_nodes_in_layer,num_outputs);
				network.giveStartingNodesJValue(hiden_nodes_in_layer);
			}
			
			for(Network network: networks)
			{
				network.adjustWeights(learning_num,hiden_nodes_in_layer);
			}
			
		}
		
		
		/*for (Network network : networks)//displays all values
		{
			ArrayList<Double> all_values = new ArrayList<Double>();
			all_values = network.getAllValues();
			for(double value : all_values)
			{
				System.out.println(value);
			}
			System.out.println("\n");
		}*/
		
		
		/*for (Network network : networks)//displays all j_values
		{
			ArrayList<Double> testing = new ArrayList<Double>();
			testing = network.getJValuesOfAllNodes();
			for(double t : testing)
			{
				System.out.println(t);
			}
			System.out.println("\n");
		}*/
	
		
		System.out.println("\n");
		System.out.println("\n");
		System.out.println("Scale: " + max);
		System.out.println("\n");
		for (Network network : networks)
		{
			int t = 0;
			ArrayList<Double> input_values = new ArrayList<Double>();
			ArrayList<Double> output_values = new ArrayList<Double>();
			ArrayList<Double> network_measured_outputs  = new ArrayList<Double>();
			input_values = network.getOGInputs();
			for(double input_value : input_values)
			{
				System.out.println("Input: " + input_value); 
			}
			network_measured_outputs = network.getMeasuredOutputs();
			output_values = network.getOutputs(num_outputs);
			for(double output_value : output_values)
			{
				System.out.println("Measured Output: " + network_measured_outputs.get(t));
				System.out.println("Neural Net Output: " + output_value * max);
				t++;
			}
			System.out.println("\n");
		}
		
		
	}
}