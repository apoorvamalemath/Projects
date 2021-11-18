#include<stdio.h>
#include<stdlib.h>


void main()
{
  int n,i,j,m;
  printf("Enter the number of memory locations\n");
  scanf("%d",&n);
  
  int mem[n][2],res=0,t;

  printf("Enter the size of each memory location\n");
  
  for(i=0;i<n;i++)
    { scanf("%d",&mem[i][0]); }

  for(i=1;i<n;i++)
     for(j=0;j<n-i;j++)
       {
          if(mem[j][0]<mem[j+1][0])
             {t=mem[j][0];
              mem[j][0]=mem[j+1][0];
              mem[j+1][0]=t;}

      }

  for(i=0;i<n;i++)
    {
      mem[i][1]=mem[i][0];  
    }

  printf("Enter the number of processes\n");
  scanf("%d",&m);
  
  int proc[m];

  printf("Enter the size of each process\n");
  
  for(i=0;i<m;i++)
    scanf("%d",&proc[i]);
  
  i=0;

 printf("mem loc  process no\n");
 while(i<m)
{
  for(j=0;j<n;j++)
   {
     if(mem[j][1]>=proc[i] )
        { printf("%d \t process %d\n",mem[j][0],i+1);  mem[j][1]=mem[j][1]-proc[i];  break;  }
   }
    i++;
}
         

  printf("    Memory size    fragmentation\n");
   for(i=0;i<n;i++)
          printf("%d  %d\t\t    %d\t\n",i+1,mem[i][0],mem[i][1]);   

  printf("Total fragmentation\n");
  
   for(i=0;i<n;i++)
      res+=mem[i][1];

  printf("%d\n",res);
        




}
