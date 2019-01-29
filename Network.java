package neuralnet;
import java.util.ArrayList;
import java.lang.Math;
public class Network
{
	ArrayList<Node> Starting_Nodes;
	ArrayList<Node> Connecting_Nodes;
	ArrayList<Double> measured_outputs;
	//ArrayList<Integer> Weights;
	public Network()
	{
		Starting_Nodes = new ArrayList<Node>();
		Connecting_Nodes = new ArrayList<Node>(); // know where they are in the graph based off placement in the AL
		measured_outputs = new ArrayList<Double>();
		//Weights = new ArrayList<Integer>();
	}
	
	public void addStaringNode(Node node)
	{
		Starting_Nodes.add(node);
	}
	
	public void addConnectedNode(Node node)
	{
		Connecting_Nodes.add(node);
	}
	
	public void addMeasuredOutputs(double value)
	{
		measured_outputs.add(value);
	}
	
	public ArrayList<Double> getMeasuredOutputs()
	{
		return measured_outputs;
	}
	
	
	public int getSizeofStartingNodes()
	{
		return Starting_Nodes.size();
	}
	
	public void randomizeConnectedNodeWeights(int which_node, int num_weights)
	{
		Connecting_Nodes.get(which_node).addRandomWeight(num_weights);
	}
	
	public ArrayList<Double> getConnectedNodeWeights(int which_node)
	{
		return Connecting_Nodes.get(which_node).getWeights();
	}
	
	public void setStartingNodesInputBack()
	{
		for(Node node: Starting_Nodes)
		{
			node.changeInput(node.getStartingInput());
		}
	}
	
	private double preformGFunction(ArrayList<Double> inputs_to_be_summed)
	{
		double sum = 0;
		double g_value = 0;
		for(double d : inputs_to_be_summed)
		{
			sum += d;
		}
		
		sum *= -1;
		g_value = 1/(1+Math.exp(sum));
		return g_value;
		
	}
	
	private double preformDGFunction(double current_input)
	{
		return current_input * (1 - current_input);
	}
	
	private void preformBackwardsDGFunction(int array_pos, ArrayList<Double> inputs_to_be_summed, boolean connectnode)
	{
		double sum = 0;
		double j_value = 0;
		double g_value = 0;
		if(connectnode)
		{
			g_value = this.preformDGFunction(Connecting_Nodes.get(array_pos).getInput());
		}
		else
		{
			g_value = this.preformDGFunction(Starting_Nodes.get(array_pos).getInput());
		}
		
		for(double d : inputs_to_be_summed)
		{
			sum += d;
		}
		j_value = g_value * sum;
		
		
		if(connectnode)
		{
			Connecting_Nodes.get(array_pos).setJValue(j_value);
		}
		else
		{
			Starting_Nodes.get(array_pos).setJValue(j_value);
		}
		
	}
	
	public ArrayList<Double> getOutputs(int num_outputs)
	{
		ArrayList<Double> outputs = new ArrayList<Double>();
		int outputs_start = Connecting_Nodes.size() - num_outputs;
		for(int i = outputs_start; i < Connecting_Nodes.size(); i++)
		{
			outputs.add(Connecting_Nodes.get(i).getInput());
		}
		return outputs;
	}
	
	public ArrayList<Double> getOGInputs()
	{
		ArrayList<Double> input_values = new ArrayList<Double>();
		for(Node input : Starting_Nodes)
		{
			input_values.add(input.getStartingInput());
		}
		return input_values;
	}
	
	
	public void giveConnectingNodesInput(int hiden_nodes_in_layer, int num_inputs)
	{
		int i = 0;
		int layer_conection = 0;
		for(Node node: Connecting_Nodes)
		{
			double g_value = 0;
			ArrayList<Double> inputs_to_be_summed = new ArrayList<Double>();
			
			if(i != hiden_nodes_in_layer && i % hiden_nodes_in_layer == 0 && i != 0) // Check to see if layer_conection needs to double to make sure values are with their weights
			{
				layer_conection = layer_conection + hiden_nodes_in_layer;
			}
			
			if (i < hiden_nodes_in_layer) // first layer of nodes
			{
				ArrayList<Double> weights = new ArrayList<Double>();
				weights = node.getWeights();
				for(int x = 0; x < weights.size(); x++)
				{
					if(x < num_inputs)
					{
						inputs_to_be_summed.add(weights.get(x)*Starting_Nodes.get(x).getStartingInput());
					}
					else
					{
						inputs_to_be_summed.add(weights.get(x)*1);//bias weight
					}
				}
				g_value = this.preformGFunction(inputs_to_be_summed);
				node.changeInput(g_value);
			}
			else // rest of the layers
			{
				ArrayList<Double> weights = new ArrayList<Double>();
				weights = node.getWeights();
				int temp = layer_conection;
				for(int x = 0; x < weights.size(); x++)
				{ 
					//int temp = layer_conection;
					if(x < hiden_nodes_in_layer)
					{
						inputs_to_be_summed.add(weights.get(x)*Connecting_Nodes.get(temp).getInput());
					}
					else
					{
						inputs_to_be_summed.add(weights.get(x)*1);//bias weight
					}
					temp = temp + 1;	
				}
				g_value = this.preformGFunction(inputs_to_be_summed);
				node.changeInput(g_value);
			}
			i++;
		}
	}
	
	public void giveOutputNodesJValue(int num_outputs, double max)
	{
		int outputs_start = Connecting_Nodes.size() - num_outputs;
		int x = 0;
		for(int i = outputs_start; i < Connecting_Nodes.size(); i++)
		{
			Node current_node = Connecting_Nodes.get(i);
			
			 double j_value = this.preformDGFunction(current_node.getInput())*(measured_outputs.get(x) - (current_node.getInput()*max));
			 //System.out.println((current_node.getInput()*max));
			 
			 current_node.setJValue(j_value);
			 x++;
		}
	}
	
	public void giveHiddenLayerNodesJValue(int hiden_nodes_in_layer,int num_outputs)
	{
		int outputs_start = Connecting_Nodes.size() - num_outputs;
		int hidden_layer_end = outputs_start - 1;
		int last_hidden_layer = hidden_layer_end - hiden_nodes_in_layer;
		int t = 1; // selects which output to get for last_hidden_layer
		int v = 2; // starts at two as to not get bias weight for last_hidden_layer
		
		int p = 0; // count so we know when to reset for new layer
		int w = hiden_nodes_in_layer;
		int l = 2; // starts at two as to not get bias weight for other_hidden_layers
		for (int i = hidden_layer_end; i >= 0 ; i--)
		{
			if(p != 0 && p % hiden_nodes_in_layer == 0)
			{
				w = hiden_nodes_in_layer;
				l = 2;
			}
			
			ArrayList<Double> inputs_to_be_summed = new ArrayList<Double>();
			if(i > last_hidden_layer)// deals with last hidden layer
			{
				for(int x = 0; x < num_outputs; x++)
				{
					Node output_node = Connecting_Nodes.get(Connecting_Nodes.size()-t);
					ArrayList<Double> output_weights = new ArrayList<Double>();
					output_weights = output_node.getWeights();
					double output_weight = output_weights.get(output_weights.size()-v);
					inputs_to_be_summed.add(output_weight*output_node.getJValue());
					t++;
					//v++;
				}
				this.preformBackwardsDGFunction(i,inputs_to_be_summed,true);
				t = 1;
				v++;
			}
			else
			{
				int r = w + i; // used to select which node to get weight from
				for(int x = 0; x < hiden_nodes_in_layer; x++)
				{
					//int r = w + i; // used to select which node to get weight from
					Node hidden_layer_node = Connecting_Nodes.get(r);
					ArrayList<Double> hidden_layer_node_weights = new ArrayList<Double>();
					hidden_layer_node_weights = hidden_layer_node.getWeights();
					double hidden_layer_node_weight = hidden_layer_node_weights.get(hidden_layer_node_weights.size()-l);
					inputs_to_be_summed.add(hidden_layer_node_weight*hidden_layer_node.getJValue());
					r--;
				}
				this.preformBackwardsDGFunction(i,inputs_to_be_summed,true);
				w++;
				l++;
			}
			p++;
		}
	}
	
	public void giveStartingNodesJValue(int hiden_nodes_in_layer)
	{
		int x = 0;
		for(Node starting_node : Starting_Nodes)
		{
			ArrayList<Double> inputs_to_be_summed = new ArrayList<Double>();
			for(int i = 0; i < hiden_nodes_in_layer; i++)
			{
				ArrayList<Double> first_layer_node_weights = new ArrayList<Double>();
				first_layer_node_weights = Connecting_Nodes.get(i).getWeights();
				double first_layer_node_weight = first_layer_node_weights.get(x);
				inputs_to_be_summed.add(first_layer_node_weight*Connecting_Nodes.get(i).getJValue());
			}
			this.preformBackwardsDGFunction(x,inputs_to_be_summed,false);
			x++;
		}
	}
	
	public void adjustWeights(float learning_rate, int hiden_nodes_in_layer)
	{
		int i = 0;
		int layer_conection = 0;
		for(Node connecting_node : Connecting_Nodes)
		{
			ArrayList<Double> weights = new ArrayList<Double>();
			weights = connecting_node.getWeights();
			double new_weight = 0;
			if(i != hiden_nodes_in_layer && i % hiden_nodes_in_layer == 0 && i != 0) // Check to see if layer_conection needs to double to make sure values are with their weights
			{
				layer_conection = layer_conection + hiden_nodes_in_layer;
			}
			
			if(i < hiden_nodes_in_layer)//deals with first layer
			{
				for(int x = 0; x < weights.size(); x++)
				{
					if(x == weights.size() - 1)
					{
						new_weight = weights.get(x) + (learning_rate * 1 * connecting_node.getJValue()); // deals with bias
					}
					else
					{
						new_weight = weights.get(x) + (learning_rate * Starting_Nodes.get(x).getInput() * connecting_node.getJValue());
					}
					connecting_node.updateWeight(new_weight,x);
				}
				
			}
			else // rest of the layers
			{
				int temp = layer_conection;
				for(int x = 0; x < weights.size(); x++)
				{ 
					//int temp = layer_conection;
					if(x < hiden_nodes_in_layer)
					{
						//inputs_to_be_summed.add(weights.get(x)*Connecting_Nodes.get(temp).getInput());
						
						new_weight = weights.get(x) + (learning_rate * Connecting_Nodes.get(temp).getInput() * connecting_node.getJValue());
					}
					else
					{
						//inputs_to_be_summed.add(weights.get(x)*1);//bias weight
						
						new_weight = weights.get(x) + (learning_rate * 1 * connecting_node.getJValue()); //bias weight
					}
					connecting_node.updateWeight(new_weight,x);
					temp = temp + 1;	
				}
			}
			i++;
		}
	
		/*for(Node connecting_node : Connecting_Nodes)
		{
			
			ArrayList<Double> weights = new ArrayList<Double>();
			weights = connecting_node.getWeights();
			for(int i = 0; i < weights.size(); i++)
			{
				double new_weight = weights.get(i) + (learning_rate * connecting_node.getInput() * connecting_node.getJValue());
				connecting_node.updateWeight(new_weight,i);
				//weights.set(i,new_weight);
			}
		}*/
	}
	
	
	public ArrayList<Double> getJValuesOfOutputs(int num_outputs)
	{
		ArrayList<Double> j_values = new ArrayList<Double>();
		int outputs_start = Connecting_Nodes.size() - num_outputs;
		for(int i = outputs_start; i < Connecting_Nodes.size(); i++)
		{
			j_values.add(Connecting_Nodes.get(i).getJValue());
		}
		return j_values;
	}
	
	public ArrayList<Double> getJValuesOfAllNodes()
	{
		ArrayList<Double> j_values = new ArrayList<Double>();
		
		for(Node starting_node : Starting_Nodes)
		{
			j_values.add(starting_node.getJValue());
		}
		
		for(Node connecting_node : Connecting_Nodes)
		{
			j_values.add(connecting_node.getJValue());
		}
		
		return j_values;
		
	}
	
	public ArrayList<Double> getAllValues()
	{
		ArrayList<Double> values = new ArrayList<Double>();
		
		for(Node starting_node : Starting_Nodes)
		{
			values.add(starting_node.getInput());
		}
		
		for(Node connecting_node : Connecting_Nodes)
		{
			values.add(connecting_node.getInput());
		}
		
		return values;
	}
	
}